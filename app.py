"""Fluxo de Caixa com IA: extrato bancário em CSV -> lançamentos categorizados, fluxo mensal, projeção e alertas."""

from typing import Literal

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from ia import ErroIA, extrair

CATEGORIAS = [
    "Receita de vendas", "Receita financeira", "Fornecedores", "Folha de pagamento", "Impostos",
    "Aluguel", "Água, luz e internet", "Marketing", "Software e assinaturas", "Tarifas bancárias",
    "Transporte e combustível", "Empréstimos e financiamentos", "Retirada dos sócios",
    "Transferência entre contas", "Outros",
]
Categoria = Literal[tuple(CATEGORIAS)]
LOTE = 25  # descrições por chamada; modelos locais pequenos erram mais com listas longas


class Classificacao(BaseModel):
    indice: int
    categoria: Categoria


class Lote(BaseModel):
    itens: list[Classificacao]


class Analise(BaseModel):
    resumo: str = Field(description="3 a 4 frases sobre a saúde do caixa")
    alertas: list[str]
    recomendacoes: list[str] = Field(description="Ações práticas, cada uma citando o número que a motiva")


INSTRUCOES_CATEGORIA = f"""Você classifica lançamentos de extrato bancário de uma pequena empresa brasileira.
Para cada descrição, escolha uma categoria desta lista: {", ".join(CATEGORIAS)}.
O sinal do valor indica se é entrada (+) ou saída (-). Repasses de maquininha e adquirente
(Cielo, Stone, Rede, GetNet, PagSeguro, Mercado Pago), inclusive antecipação, são receita de
vendas; receita financeira é só rendimento de aplicação e juros recebidos.
Pagamento a quem vende mercadoria ou insumo (distribuidora, atacado, torrefação, fábrica,
fornecedor) é Fornecedores. Impostos é só tributo: DAS, DARF, ICMS, ISS, INSS, FGTS, IPTU, IPVA.
Devolva um item para cada índice recebido."""

INSTRUCOES_ANALISE = """Você é analista financeiro de pequenas empresas. Recebe o fluxo de caixa mensal já
calculado, os gastos por categoria e uma projeção simples. Explique a situação do caixa em
linguagem de dono de empresa, aponte alertas (queda de receita, gasto crescendo, caixa ficando
negativo, concentração de despesas) e recomende ações. Use só os números e porcentagens
recebidos, escritos em reais (R$) ou %: não calcule porcentagens novas."""


def ler_extrato(arquivo) -> pd.DataFrame:
    df = pd.read_csv(arquivo, sep=None, engine="python", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    coluna = lambda *nomes: next(c for c in df.columns if any(n in c for n in nomes))
    try:
        data, desc, valor = coluna("data"), coluna("descri", "hist", "lanc"), coluna("valor", "quantia")
    except StopIteration:
        raise ErroIA("O CSV precisa ter colunas de data, descrição e valor.")
    valores = df[valor].str.replace("R$", "", regex=False).str.strip()
    if valores.str.contains(",").any():  # formato brasileiro: 1.234,56
        valores = valores.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.DataFrame({
        "data": pd.to_datetime(df[data], dayfirst=True, errors="coerce"),
        "descricao": df[desc].str.strip(),
        "valor": pd.to_numeric(valores, errors="coerce"),
    }).dropna()


def categorizar(df: pd.DataFrame, avancar) -> pd.DataFrame:
    # classifica cada descrição única uma vez só (ex.: "TARIFA PACOTE" aparece todo mês)
    unicos = df.groupby("descricao", as_index=False)["valor"].mean()
    rotulos: dict[str, str] = {}
    for inicio in range(0, len(unicos), LOTE):
        lote = unicos.iloc[inicio:inicio + LOTE].reset_index(drop=True)
        texto = "\n".join(f"{i}. {r.descricao} | {r.valor:+.2f}" for i, r in lote.iterrows())
        for item in extrair(texto, Lote, INSTRUCOES_CATEGORIA).itens:
            if 0 <= item.indice < len(lote):
                rotulos[lote.descricao[item.indice]] = item.categoria
        avancar(min(inicio + LOTE, len(unicos)) / len(unicos))
    df = df.copy()
    df["categoria"] = df.descricao.map(lambda d: rotulos.get(d, "Outros"))
    df["natureza"] = natureza(df)
    return df


def natureza(df: pd.DataFrame) -> pd.Series:
    """Fixo = saída que aparece em todos os meses com valor estável (variação de até 15%). Regra no código, não na IA."""
    meses = df.data.dt.to_period("M")
    por_mes = df[df.valor < 0].groupby([df.descricao, meses]).valor.sum().abs()
    estatisticas = por_mes.groupby(level=0).agg(["count", "mean", "std"]).fillna(0)
    fixas = estatisticas[(estatisticas["count"] == meses.nunique()) & (estatisticas["std"] <= 0.15 * estatisticas["mean"])].index
    return pd.Series(["fixo" if (v < 0 and d in fixas) else "variável" for d, v in zip(df.descricao, df.valor)], index=df.index)


def fluxo_mensal(df: pd.DataFrame, saldo_inicial: float = 0) -> pd.DataFrame:
    sem_transferencia = df[df.categoria != "Transferência entre contas"]
    mes = sem_transferencia.data.dt.to_period("M")
    fluxo = pd.DataFrame({
        "entradas": sem_transferencia.valor.clip(lower=0).groupby(mes).sum(),
        "saidas": -sem_transferencia.valor.clip(upper=0).groupby(mes).sum(),
    })
    fluxo["resultado"] = fluxo.entradas - fluxo.saidas
    fluxo["saldo_acumulado"] = saldo_inicial + fluxo.resultado.cumsum()
    fluxo.index = fluxo.index.astype(str)
    return fluxo


def projecao(fluxo: pd.DataFrame, meses: int = 3) -> pd.DataFrame:
    """Média dos últimos 3 meses projetada para frente. Simples de propósito: fácil de explicar."""
    base = fluxo.tail(3)[["entradas", "saidas"]].mean()
    ultimo = pd.Period(fluxo.index[-1], "M")
    saldo = fluxo.saldo_acumulado.iloc[-1]
    linhas = []
    for i in range(1, meses + 1):
        saldo += base.entradas - base.saidas
        linhas.append({"mes": str(ultimo + i), "entradas": base.entradas, "saidas": base.saidas,
                       "resultado": base.entradas - base.saidas, "saldo_acumulado": saldo})
    return pd.DataFrame(linhas).set_index("mes")


def resumo_para_ia(fluxo: pd.DataFrame, por_categoria: pd.Series, fixos: float, futuro: pd.DataFrame) -> str:
    """Todas as contas (variações e participações) ficam no código; o modelo só interpreta."""
    mensal = fluxo.assign(
        variacao_entradas=fluxo.entradas.pct_change().map(lambda v: f"{v:+.1%}" if pd.notna(v) else "-"),
        variacao_saidas=fluxo.saidas.pct_change().map(lambda v: f"{v:+.1%}" if pd.notna(v) else "-"),
    )
    categorias = pd.DataFrame({"total": por_categoria.round(2),
                               "participacao": (por_categoria / por_categoria.sum()).map(lambda v: f"{v:.1%}")})
    return (f"<fluxo_mensal>\n{mensal.round(2).to_csv()}</fluxo_mensal>\n"
            f"<saidas_por_categoria>\n{categorias.to_csv()}</saidas_por_categoria>\n"
            f"<custo_fixo_medio_mensal>{fixos:.2f}</custo_fixo_medio_mensal>\n"
            f"<projecao_3_meses_media_dos_ultimos_3>\n{futuro.round(2).to_csv()}</projecao_3_meses_media_dos_ultimos_3>")


def reais(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


st.set_page_config(page_title="Fluxo de Caixa com IA", page_icon="💰", layout="wide")
st.title("💰 Fluxo de Caixa com IA")
st.caption("Envie o extrato bancário em CSV. A IA categoriza cada lançamento e o app monta o fluxo de caixa, a projeção e os alertas.")

esquerda, direita = st.columns([3, 1])
arquivo = esquerda.file_uploader("Extrato (.csv com data, descrição e valor)", type=["csv"])
saldo_inicial = direita.number_input("Saldo em conta no início do extrato (R$)", value=30000.0, step=1000.0)

if arquivo and st.button("Categorizar lançamentos", type="primary"):
    try:
        extrato = ler_extrato(arquivo)
        barra = st.progress(0.0, text="Categorizando…")
        st.session_state["lancamentos"] = categorizar(extrato, lambda p: barra.progress(p, text="Categorizando…"))
        st.session_state.pop("analise", None)
    except ErroIA as e:
        st.error(str(e))

if (lancamentos := st.session_state.get("lancamentos")) is not None:
    st.markdown("### Lançamentos")
    st.caption("Confira as categorias e corrija o que precisar: os números abaixo são recalculados na hora.")
    lancamentos = st.data_editor(
        lancamentos, use_container_width=True, hide_index=True, height=280,
        column_config={
            "data": st.column_config.DateColumn(format="DD/MM/YYYY", disabled=True),
            "valor": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "categoria": st.column_config.SelectboxColumn(options=CATEGORIAS, required=True),
            "natureza": st.column_config.SelectboxColumn(options=["fixo", "variável"], required=True),
        },
    )

    fluxo = fluxo_mensal(lancamentos, saldo_inicial)
    futuro = projecao(fluxo)
    ultimo = fluxo.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entradas no último mês", reais(ultimo.entradas))
    c2.metric("Saídas no último mês", reais(ultimo.saidas))
    c3.metric("Resultado do mês", reais(ultimo.resultado))
    c4.metric("Saldo em conta projetado em 3 meses", reais(futuro.saldo_acumulado.iloc[-1]))
    if (negativos := futuro[futuro.saldo_acumulado < 0]).shape[0]:
        st.error(f"No ritmo atual o caixa fica negativo em **{negativos.index[0]}**.")

    esquerda, direita = st.columns(2)
    esquerda.markdown("#### Entradas x saídas por mês")
    esquerda.bar_chart(fluxo[["entradas", "saidas"]], stack=False, color=["#2e9e5b", "#d9534f"])
    direita.markdown("#### Saldo acumulado (real + projeção)")
    saldo = pd.concat([fluxo.saldo_acumulado.rename("real"), futuro.saldo_acumulado.rename("projeção")], axis=1)
    direita.line_chart(saldo)

    saidas = lancamentos[lancamentos.valor < 0]
    por_categoria = (-saidas.groupby("categoria").valor.sum()).sort_values(ascending=False)
    fixos = -saidas[saidas.natureza == "fixo"].valor.sum() / max(len(fluxo), 1)
    st.markdown("#### Para onde vai o dinheiro")
    st.bar_chart(por_categoria, horizontal=True)
    st.caption(f"Custo fixo médio por mês: **{reais(fixos)}**")

    if st.button("Analisar com IA"):
        with st.spinner("Analisando o caixa…"):
            resumo = resumo_para_ia(fluxo, por_categoria, fixos, futuro)
            try:
                st.session_state["analise"] = extrair(resumo, Analise, INSTRUCOES_ANALISE)
            except ErroIA as e:
                st.error(str(e))

    if analise := st.session_state.get("analise"):
        st.info(analise.resumo)
        a, b = st.columns(2)
        a.markdown("**🚨 Alertas**\n" + "\n".join(f"- {x}" for x in analise.alertas))
        b.markdown("**✅ Recomendações**\n" + "\n".join(f"- {x}" for x in analise.recomendacoes))

    st.download_button("Baixar lançamentos categorizados (.csv)",
                       lancamentos.to_csv(index=False, sep=";", decimal=","), "lancamentos.csv", "text/csv")
