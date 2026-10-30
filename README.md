# 💰 Fluxo de Caixa com IA

Recebe o **extrato bancário em CSV** e entrega o **fluxo de caixa pronto**: cada lançamento categorizado pela IA (fornecedores, folha, impostos, aluguel…), entradas x saídas por mês, para onde vai o dinheiro, custo fixo mensal, **projeção de 3 meses** com aviso do mês em que o caixa fica negativo, e uma análise com alertas e recomendações.

## O problema

Pequena empresa costuma descobrir que o caixa vai faltar quando ele já faltou. Montar o fluxo de caixa exige classificar centenas de lançamentos do extrato na mão, todo mês.

## Como a IA resolve

O desenho separa o que é **julgamento** (IA) do que é **conta** (código):

| Etapa | Quem faz |
|---|---|
| Ler o CSV (aceita `;` ou `,`, valor em formato brasileiro) | código (pandas) |
| Categorizar cada descrição do extrato | **IA**, em lotes, cada descrição única uma vez só |
| Revisar a categoria | **você**, direto na tabela; tudo é recalculado na hora |
| Fixo x variável | código: saída que aparece todo mês com variação de até 15% |
| Fluxo mensal, variações, participação por categoria e projeção | código (pandas) |
| Explicar a situação, alertas e recomendações | **IA**, recebendo só os números já calculados |

Assim a IA não erra conta, e o resultado é auditável. Tudo roda **100% local**: o extrato bancário não sai da máquina.

## Rodando

Precisa do [Ollama](https://ollama.com) instalado e aberto.

```bash
ollama pull llama3.1:8b
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Para testar, envie [`exemplos/extrato-loja.csv`](exemplos/extrato-loja.csv): 3 meses de uma loja com receita caindo, e o app mostra em que mês o caixa fica negativo. O CSV só precisa de colunas de data, descrição e valor (negativo = saída), que é o formato que os bancos exportam.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `app.py` | Leitura do extrato, categorização em lotes, cálculos do fluxo, gráficos e exportação |
| `ia.py` | Chamada ao Ollama com saída estruturada e tratamento de erros |

## Stack

Python · Streamlit · pandas · Ollama (Llama 3.1, local, structured outputs) · Pydantic

## Próximos passos

- Ler OFX além de CSV
- Guardar as categorias corrigidas para o modelo não errar de novo no mês seguinte
