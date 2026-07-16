# Instruções do projeto AiStudents

Este repositório implementa um pipeline de aprendizado de máquina para prever a evolução do desempenho acadêmico de estudantes com base em uso de IA generativa, hábitos de estudo e características do perfil do aluno.

## Objetivo do projeto

- Treinar uma rede neural MLP para prever a variação de GPA ($\Delta$ GPA) a partir de dados tabulares.
- Simular cenários pedagógicos de uso de IA e estudar o impacto de intervenções.
- Gerar visualizações explicativas em arquivos da pasta plots.

## Estrutura principal

- main.py: ponto central do pipeline completo.
  - Carrega o dataset.
  - Realiza pré-processamento.
  - Divide os dados em treino e validação.
  - Treina a rede neural.
  - Gera gráficos e salva o modelo.
- plots.py: funções para gerar gráficos de análise e sensibilidade.
- data/ai_student_impact_dataset.csv: dataset principal do projeto.
- simuvered_model.pth: modelo treinado salvo em disco.

## Fluxo de execução

1. Verificar se o arquivo CSV existe em data/.
2. Executar main.py para rodar o pipeline completo.
3. Confirmar que os gráficos sejam salvos em plots/.
4. Garantir que o modelo treinado seja salvo como simuvered_model.pth.

## Regras de desenvolvimento

- Escrever código e comentários em português, sempre que possível.
- Manter a compatibilidade com PyTorch, pandas, NumPy, scikit-learn e matplotlib.
- Preservar a reprodutibilidade do experimento usando sementes fixas.
- Evitar mudanças bruscas na estrutura do dataset sem ajustar o pré-processamento.
- Se novas colunas forem adicionadas ou renomeadas, atualizar também as features usadas no pré-processador e nos exemplos de estudante.
- Preferir soluções simples, legíveis e bem documentadas.

## Convenções importantes

- O alvo do modelo é a coluna Delta_GPA_norm, calculada a partir da diferença entre Post_Semester_GPA e Pre_Semester_GPA.
- O pré-processamento usa StandardScaler para variáveis numéricas e OneHotEncoder para variáveis categóricas.
- O modelo é uma MLP com camadas lineares e Dropout/BatchNorm.
- Ao criar novas funções, manter o estilo do projeto e nomear de forma explícita.

## Boas práticas para alterações

- Testar o fluxo completo após alterações em preprocessing, treino ou gráficos.
- Se alterar o nome de variáveis ou arquivos, manter consistência com o código existente.
- Para novos gráficos, salvar em plots/ e manter o mesmo padrão visual do projeto.

## Linguagem esperada

- Responder em português brasileiro.
- Explicar alterações de forma clara, com foco em execução, análise e manutenção do projeto.
