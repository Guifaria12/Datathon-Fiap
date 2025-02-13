#Importação das bibliotecas
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['Ano ingresso', 'origem']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class MinMax(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler  = ['Fase', 'IAA', 'IEG', 'IPS', 'IDA', 'Matem', 'Portug', 'Inglês', 'IPV', 'IAN', 'Idade', 'IPP', 'Defasagem', 'INDE']):
        self.min_max_scaler = min_max_scaler
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler ).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler] = min_max_enc.fit_transform(df[self.min_max_scaler ])
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class OneHotEncodingNames(BaseEstimator,TransformerMixin):
    def __init__(self,OneHotEncoding = ['Gênero', 'Instituição de ensino']):

        self.OneHotEncoding = OneHotEncoding

    def fit(self,df):
        return self

    def transform(self,df):
        if (set(self.OneHotEncoding).issubset(df.columns)):
            # função para one-hot-encoding das features
            def one_hot_enc(df,OneHotEncoding):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[OneHotEncoding])
                # obtendo o resultado dos nomes das colunas
                feature_names = one_hot_enc.get_feature_names_out(OneHotEncoding)
                # mudando o array do one hot encoding para um dataframe com os nomes das colunas
                df = pd.DataFrame(one_hot_enc.transform(df[self.OneHotEncoding]).toarray(),
                                  columns= feature_names,index=df.index)
                return df

            # função para concatenar as features com aquelas que não passaram pelo one-hot-encoding
            def concat_with_rest(df,one_hot_enc_df,OneHotEncoding):
                # get the rest of the features
                outras_features = [feature for feature in df.columns if feature not in OneHotEncoding]
                # concaternar o restante das features com as features que passaram pelo one-hot-encoding
                df_concat = pd.concat([one_hot_enc_df, df[outras_features]],axis=1)
                return df_concat

            # one hot encoded dataframe
            df_OneHotEncoding = one_hot_enc(df,self.OneHotEncoding)

            # retorna o dataframe concatenado
            df_full = concat_with_rest(df, df_OneHotEncoding,self.OneHotEncoding)
            return df_full

        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='Status_entrada'):
        self.target_col = target_col  # Define a coluna alvo

    def fit(self, df, y=None):
        return self  # Como não treinamos nada, apenas retorna a instância

    def transform(self, df):
        if self.target_col in df.columns:
            oversample = SMOTE(sampling_strategy='minority')

            # Separar features e target
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]

            # Aplicar SMOTE
            X_bal, y_bal = oversample.fit_resample(X, y)

            # Criar novo DataFrame balanceado
            df_bal = pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name=self.target_col)], axis=1)

            return df_bal
        else:
            raise ValueError(f"A coluna alvo '{self.target_col}' não está no DataFrame.")

class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Mau' in df.columns:
            # função smote para superamostrar a classe minoritária para corrigir os dados desbalanceados
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Status_entrada'], df['Status_entrada'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("O target não está no DataFrame")
            return df

from sklearn.pipeline import Pipeline

def pipeline(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('min_max_scaler', MinMax()),
        ('oversample', Oversample())
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

base_completa = pd.read_csv("df_clean.csv")

############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)
st.subheader ("""
**Faculdade de Informática e Administração Paulista – FIAP**

**Grupo 47: Cristiano de Araujo Santos Filho, Eduardo Vilela Silva, Guilherme de Faria
Rodrigues, Marcelo Pereira Varesi e Vitória Pinheiro de Mattos**
""")
st.header("**Introdução**")

st.write("""
A educação é um fator crucial para o desenvolvimento social e econômico, especialmente em comunidades vulneráveis. A ONG "Passos Mágicos" atua nesse cenário, oferecendo suporte educacional e socioeconômico para crianças e jovens, com o objetivo de transformar suas realidades. Este trabalho busca contribuir para essa missão por meio da análise de dados e da construção de modelos preditivos que avaliem o impacto das ações da organização.

Com base em dados coletados entre 2020 e 2022, que incluem informações educacionais e socioeconômicas, propõe-se uma análise descritiva para identificar padrões e indicadores de desempenho dos estudantes. Além disso, será desenvolvido um modelo preditivo utilizando técnicas de machine learning ou deep learning para prever o comportamento dos estudantes com base em variáveis-chave, como desempenho acadêmico e condições socioeconômicas.

O objetivo final é fornecer insights que possam auxiliar a "Passos Mágicos" na tomada de decisões estratégicas, ampliando o impacto positivo de suas iniciativas. Através da ciência de dados, busca-se contribuir para a transformação educacional e social das comunidades atendidas pela organização.""")

st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> DATATHON - Modelo Preditivo </h1>", unsafe_allow_html = True)

st.warning('Preencha o formulário com todos os seus dados e clique no botão **ENVIAR** no final da página.')

st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)
# Cria duas colunas
col1, col2 = st.columns(2)

# Usando um container para o conteúdo
with st.container():
    # Fase na primeira coluna dentro do container
    with col1:
        st.write('### Fase')
        input_fase = float(st.slider('Selecione a sua nota', 0, 8, key='nota_fase'))

# IEG na segunda coluna
with col2:
    st.write('### Defasagem')
    input_defasagem = float(st.slider('Selecione a sua nota', -5, 5, key='nota_defasagem'))

st.warning("""
- **Fase:** Fase do Aluno na Passos Mágicos que está relacionado ao Nível de Aprendizado (de 0 (Alfa) até 8)  
- **Defasagem:** Mostra o nível de defasagem do ano 
""")

st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

# Cria duas colunas
col1, col2, col3 = st.columns(3)

# Usando um container para o conteúdo
with st.container():
    # IAA na primeira coluna dentro do container
    with col1:
        st.write('### IAA')
        input_iaa = float(st.slider('Selecione a sua nota', 0, 10, key='nota_iaa'))

# IEG na segunda coluna
with col2:
    st.write('### IEG')
    input_ieg = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ieg'))

# IPS na terceira coluna
with col3:
     st.write('### IPS')
     input_ips = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ips'))

# Cria a segunda linha com 1 coluna para a quarta coluna
col4, col5, col6, col7 = st.columns(4) # Cria a quarta coluna abaixo da primeira

with col4:
    st.write('### IDA')
    input_ida = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ida'))
    
# Coloca conteúdo na quarta coluna
with col5:
   st.write('### IPV')
   input_ipv = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ipv'))

with col6:
    st.write('### IAN')
    input_ian = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ian'))

with col7:
    st.write('### IPP')
    input_ipp = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ipp'))

st.warning("""
**Indicadores de Avaliação**  

- **IAN:** Indicador de Adequação ao Nível – Média das Notas de Adequação do Aluno  
- **IDA:** Indicador de Aprendizagem - Média das Notas do Indicador de Aprendizagem (de 0 até 10 pontos)  
- **IEG:** Indicador de Engajamento – Média das Notas de Engajamento do Aluno (de 0 até 10 pontos)  
- **IAA:** Indicador de Auto Avaliação – Média das Notas de Auto Avaliação do Aluno (de 0 até 10 pontos)  

**Indicadores de Conselho**  

- **IPS:** Indicador Psicossocial – Média das Notas Psicossociais do Aluno (de 0 até 10 pontos)  
- **IPP:** Indicador Psicopedagógico – Média das Notas Psico Pedagógicas do Aluno (de 0 até 10 pontos)  
- **IPV:** Indicador de Ponto de Virada – Média das Notas de Ponto de Virada do Aluno (de 0 até 10 pontos)  
""")



st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

# Cria duas colunas
col1, col2, col3 = st.columns(3)

# Usando um container para o conteúdo
with st.container():
    # Matem na primeira coluna dentro do container
    with col1:
        st.write('### Matemática')
        input_matem = float(st.slider('Selecione a sua nota', 0, 10, key='nota_matem'))

# Portug na segunda coluna
with col2:
    st.write('### Português')
    input_portug = float(st.slider('Selecione a sua nota', 0, 10, key='nota_portug'))

with col3:
    st.write('### Inglês')
    input_ingles = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ingles'))

st.warning("""
- **Matemática:** Média das Notas de Matemática  
- **Português:** Média das Notas de Português do Aluno 
- **Inglês:** Média das Notas de Inglês do Aluno 
""")
st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

# Gênero
st.write('### Gênero')
input_genero = st.selectbox('Qual é o seu gênero?', base_completa['Gênero'].unique())

# Idade
st.write('### Idade')
input_idade = float(st.slider('Selecione a sua idade', 0, 100))

# Instituição de Ensino
st.write('### Instituição de Ensino')
input_ensino = st.selectbox('Qual é a sua instituição de ensino?', base_completa['Instituição de ensino'].unique())

def calcular_inde(df):
    def calcular_linha(row):
        if row['Fase'] == 8:
            return (row['IAN'] * 0.1 + row['IDA'] * 0.4 + row['IEG'] * 0.2 +
                    row['IAA'] * 0.1 + row['IPS'] * 0.2)
        else:
            return (row['IAN'] * 0.1 + row['IDA'] * 0.2 + row['IEG'] * 0.2 +
                    row['IAA'] * 0.1 + row['IPS'] * 0.1 + row['IPP'] * 0.1 + row['IPV'] * 0.2)

    return df.apply(calcular_linha, axis=1)

novo_cliente = [
                    0,
                    input_fase,
                    input_genero,
                    input_ensino,
                    input_iaa,
                    input_ieg,
                    input_ips,
                    input_ida,
                    input_matem,
                    input_portug,
                    input_ingles,
                    input_ipv,
                    input_ian,
                    input_idade,
                    input_ipp,
                    input_defasagem,
                    0,
                    0
                    ]

def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(base_completa, 0.2)

cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)
cliente_predict_df = cliente_predict_df.set_index(cliente_predict_df.columns[0])
cliente_predict_df['INDE'] = calcular_inde(cliente_predict_df)

teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)
teste_novo_cliente = pipeline(teste_novo_cliente)

cliente_pred = teste_novo_cliente

if st.button('Enviar'):
    model = joblib.load('logistico.joblib')
    probabilidades = model.predict_proba(cliente_pred)[-1]
    st.write(f"Probabilidade de ser veterano: {probabilidades[0]:.2f}")
    st.write(f"Probabilidade de ser desistente: {probabilidades[1]:.2f}")
        
st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

st.header("**Conclusão**")

st.write("""
A INSERIR """)
