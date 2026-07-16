# Documentação do Projeto AiStudents

## 1. Visão geral

O projeto AiStudents tem como objetivo construir um pipeline de aprendizado de máquina para prever a evolução do desempenho acadêmico de estudantes com base em características como uso de IA generativa, hábitos de estudo, perfil acadêmico e nível de ansiedade.

A solução utiliza uma rede neural do tipo MLP (MultiLayer Perceptron) para estimar a variação de GPA ao longo de um semestre, representada pela coluna Delta_GPA_norm. Além disso, o código gera gráficos explicativos e simula cenários pedagógicos para analisar o impacto do uso consciente ou inadequado de ferramentas de IA.

---

## 2. Objetivo do sistema

O projeto busca responder perguntas como:

- Como o uso de IA generativa influencia o desempenho acadêmico?
- O uso excessivo de IA, sem estratégia pedagógica, pode prejudicar o aprendizado?
- O uso equilibrado de IA com estudo tradicional pode melhorar o rendimento?

Para isso, o fluxo do projeto inclui:

1. Carregamento e preparação dos dados.
2. Treinamento de uma rede neural regressora.
3. Geração de gráficos de análise.
4. Simulação de cenários pedagógicos.
5. Salvamento do modelo treinado para uso posterior.

---

## 3. Estrutura de arquivos

### Arquivos principais

- main.py: arquivo central do pipeline completo.
- plots.py: módulo com funções para geração de gráficos.
- data/ai_student_impact_dataset.csv: dataset utilizado para treino e análise.
- simuvered_model.pth: modelo treinado salvo em disco.
- DOCUMENTACAO.md: documentação explicativa do projeto.

### Organização lógica do código

O fluxo do projeto pode ser dividido em quatro partes principais:

- Pré-processamento dos dados.
- Construção do dataset para o PyTorch.
- Definição da arquitetura da rede neural.
- Treinamento, avaliação e geração de gráficos.

---

## 4. Pré-processamento de dados

A função preprocess_data é responsável por preparar o dataset para o treinamento.

### Etapas realizadas

1. Leitura do arquivo CSV.
2. Criação da variável alvo.
3. Normalização da variável alvo.
4. Definição das colunas categóricas, numéricas e booleanas.
5. Conversão de colunas booleanas para valores inteiros.
6. Aplicação de transformações com ColumnTransformer.
7. Retorno das features transformadas e da variável alvo.

### Variável alvo

A variável alvo é criada a partir da diferença entre:

- Post_Semester_GPA
- Pre_Semester_GPA

Essa diferença é armazenada em Delta_GPA. Em seguida, ela é normalizada para reduzir a escala e facilitar o treinamento.

### Features utilizadas

As features são separadas em:

- Numéricas: horas semanais de uso de IA, horas de estudo tradicional, diversidade de ferramentas, dependência percebida, ansiedade e GPA anterior.
- Categóricas: categoria do curso, ano de estudo, caso de uso principal e habilidade com prompts.
- Booleanas: presença de assinatura paga.

### Transformações aplicadas

- Variáveis numéricas passam por StandardScaler para padronizar os valores.
- Variáveis categóricas passam por OneHotEncoder para transformar categorias em representação numérica.
- Variáveis booleanas são mantidas sem grandes transformações.

---

## 5. Dataset customizado para PyTorch

A classe StudentDataset encapsula os dados em um formato compatível com o PyTorch.

### Função da classe

Ela recebe:

- X: matriz de features já processada.
- y: vetor com a variável alvo.

Em seguida, converte esses dados para tensores do PyTorch, permitindo que o modelo consiga consumir os exemplos durante o treinamento.

### Benefício dessa abordagem

- Facilita o uso com DataLoader.
- Permite treinamento em lote.
- Mantém o código organizado e modular.

---

## 6. Arquitetura da rede neural

A classe SimuVeredRegressor define a MLP usada no projeto.

### Estrutura da rede

A rede contém camadas lineares intercaladas com:

- BatchNorm1d
- ReLU
- Dropout

Essa combinação ajuda a:

- estabilizar o treinamento,
- evitar overfitting,
- melhorar a capacidade de generalização do modelo.

### Comportamento da rede

A rede recebe como entrada um vetor com as features processadas e retorna uma previsão contínua para o valor de GPA, ou seja, realiza uma regressão.

---

## 7. Processo de treinamento

A função train_model realiza o treinamento da rede neural.

### O que acontece durante o treinamento

1. A função cria a função de perda.
2. Define o otimizador.
3. Configura um scheduler para reduzir a taxa de aprendizado quando necessário.
4. Percorre as épocas de treinamento.
5. Calcula a perda em treino e validação.
6. Salva o melhor estado do modelo baseado na validação.

### Estratégias usadas

- SmoothL1Loss: função de perda robusta para tarefas de regressão.
- AdamW: otimizador com regularização por peso.
- Early stopping: a execução pode parar antecipadamente se a validação não melhorar por várias épocas.

### Objetivo

O treinamento tem como foco encontrar um modelo que generalize bem para dados não vistos, em vez de apenas memorizar os dados de treino.

---

## 8. Simulação pedagógica de impacto

A função simulate_veredai_impact simula dois cenários para um estudante:

- cenário sem intervenção pedagógica,
- cenário com uma intervenção inspirada no uso de tutoria e estratégias de aprendizagem assistida por IA.

### Cenário sem tutor

Neste cenário, o estudante:

- possui baixa habilidade com prompts,
- usa IA de forma passiva,
- dedica poucas horas ao estudo tradicional,
- tem um uso intenso de IA sem reflexão crítica.

### Cenário com intervenção

Neste cenário, o estudante:

- melhora a habilidade com prompts,
- usa IA de forma mais ativa e orientada,
- reduz o uso excessivo de IA,
- aumenta o estudo tradicional.

### Valor da simulação

A função aplica o pré-processador aos dois cenários e usa o modelo treinado para prever a evolução do GPA em cada caso. Isso permite comparar o impacto percebido de uma intervenção pedagógica.

---

## 9. Geração de gráficos

O módulo plots.py é responsável por gerar visualizações científicas e explicativas.

### Gráficos disponíveis

#### 1. Curva de perda

Mostra a evolução da perda durante o treinamento para treino e validação.

#### 2. Real vs. previsto

Compara os valores reais de GPA com os valores previstos pelo modelo.

#### 3. Análise de sensibilidade

Mostra como a variação de horas de uso de IA afeta a previsão do GPA em diferentes perfis de estudante.

#### 4. Heatmap pedagógico

Exibe a interação entre horas de IA e horas de estudo tradicional para mostrar como diferentes combinações influenciam a previsão.

### Importância dos gráficos

Esses gráficos ajudam a interpretar o modelo, validar sua capacidade e comunicar os resultados de maneira visual.

---

## 10. Fluxo principal do script

Ao executar main.py, o programa realiza os seguintes passos:

1. Verifica se o dataset existe.
2. Carrega e prepara os dados.
3. Divide os dados entre treino e validação.
4. Cria o DataLoader para o treinamento.
5. Inicializa a rede neural.
6. Treina o modelo.
7. Gera os gráficos.
8. Executa a simulação pedagógica.
9. Salva o modelo treinado em simuvered_model.pth.

---

## 11. Reprodutibilidade

O projeto utiliza sementes fixas para garantir que os resultados sejam mais consistentes entre execuções.

Isso é importante porque:

- redes neurais podem ter variação devido à inicialização aleatória,
- a reprodutibilidade facilita a comparação entre versões do código.

---

## 12. Pontos de atenção para manutenção

Ao alterar o código, é importante considerar:

- novas colunas no dataset exigem atualização no pré-processamento,
- alterações nas features podem afetar a arquitetura e o treinamento,
- mudanças nos gráficos devem preservar a lógica visual já adotada,
- o modelo salvo deve continuar compatível com as features esperadas.

---

## 13. Como usar o projeto

### Requisitos

O projeto depende de bibliotecas como:

- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

### Execução

Para executar o pipeline completo, basta rodar:

```bash
python main.py
```

Isso gerará os gráficos na pasta plots e salvará o modelo em simuvered_model.pth.

---

## 14. Resumo executivo

O AiStudents é um projeto que une dados educacionais, redes neurais e análise pedagógica para estudar como o uso de IA generativa influencia o desempenho acadêmico. O código é organizado em etapas claras de processamento, treinamento, visualização e simulação, o que facilita tanto o estudo do modelo quanto a manutenção do projeto.
