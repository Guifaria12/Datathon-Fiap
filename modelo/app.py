#Importação das bibliotecas
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

#carregando os dados 
url_2022 = 'https://docs.google.com/spreadsheets/d/1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=csv&gid=90992733'
url_2023 = 'https://docs.google.com/spreadsheets/d/1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=csv&gid=555005642'
url_2024 = 'https://docs.google.com/spreadsheets/d/1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=csv&gid=215885893'

base_dados_2022 = pd.read_csv(url_2022)
base_dados_2023 = pd.read_csv(url_2023)
base_dados_2024 = pd.read_csv(url_2024)

base_dados_2022['origem'] = 'Base2022'
base_dados_2023['origem'] = 'Base2023'
base_dados_2024['origem'] = 'Base2024'

base_completa = pd.concat([base_dados_2022, base_dados_2023, base_dados_2024], ignore_index=True)

base_completa.sort_values(by='origem', inplace=True, ascending=False)

base_completa.drop_duplicates(keep='first', inplace=True)

base_completa.loc[base_completa['Ano ingresso'] == 2024, 'Status_entrada'] = 'Novato'

base_completa.loc[(base_completa['Ano ingresso'] != 2024) & (base_completa['origem'] == 'Base2024') , 'Status_entrada'] = 'Veterano'

base_completa['Status_entrada'].fillna('Desistente', inplace=True)

base_completa.replace('#DIV/0!', np.nan, inplace=True)
base_completa.replace('INCLUIR', np.nan, inplace=True)

print(base_completa.notnull().sum().sort_values(ascending=False).to_string())

base_completa = base_completa.drop(columns=['Nº Av', 'RA', 'Avaliador1', 'Avaliador2', 'Data de Nasc', 'Nome Anonimizado', 'Fase Ideal', 'Avaliador3', 'Ativo/ Inativo', 'Ativo/ Inativo.1', 'Escola', 'Destaque IDA', 'Destaque IPV', 'Avaliador4', 'Nome', 'Destaque IEG', 'Rec Av1', 'Fase ideal', 'Atingiu PV', 'Indicado', 'Ano nasc', 'Cg', 'Cf', 'Avaliador3', 'Rec Psicologia' ,'Ct', 'Rec Av3' , 'Rec Av2', 'Turma', 'Data de Nasc', 'Avaliador6', 'Destaque IPV.1', 'Avaliador5', 'Rec Av4'])

# Substituir as vírgulas por pontos e converter para float
base_completa[['Ing', 'Inglês']] = base_completa[['Ing', 'Inglês']].replace({',': '.'}, regex=True).astype(float)
base_completa['Inglês'] = base_completa[['Ing', 'Inglês']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Ing'], inplace=True)

# Substituir as vírgulas por pontos e converter para float
base_completa[['Mat', 'Matem']] = base_completa[['Mat', 'Matem']].replace({',': '.'}, regex=True).astype(float)
base_completa['Matem'] = base_completa[['Mat', 'Matem']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Mat'], inplace=True)

# Substituir as vírgulas por pontos e converter para float
base_completa[['Por', 'Portug']] = base_completa[['Por', 'Portug']].replace({',': '.'}, regex=True).astype(float)
base_completa['Portug'] = base_completa[['Por', 'Portug']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Por'], inplace=True)

base_completa.loc[base_completa['origem'] == 'Base2024', 'Pedra'] = base_completa['Pedra 2024']
base_completa.loc[base_completa['origem'] == 'Base2023', 'Pedra'] = base_completa['Pedra 23']
base_completa.loc[base_completa['origem'] == 'Base2022', 'Pedra'] = base_completa['Pedra 22']

base_completa.drop(columns=['Pedra 20', 'Pedra 21', 'Pedra 2024', 'Pedra 23', 'Pedra 22', 'Pedra 2023'], inplace=True)

base_completa.loc[base_completa['origem'] == 'Base2024', 'INDE'] = base_completa['INDE 2024']
base_completa.loc[base_completa['origem'] == 'Base2023', 'INDE'] = base_completa['INDE 2023']
base_completa.loc[base_completa['origem'] == 'Base2022', 'INDE'] = base_completa['INDE 22']

base_completa.drop(columns=['INDE 22', 'INDE 2023', 'INDE 23', 'INDE 2024'], inplace=True)

# Função para calcular a idade a partir de uma data no formato dd/mm/aaaa
def calcular_idade(data, ano_referencia):
    if isinstance(data, str):  # Verifica se o valor é uma string
        try:
            # Converte a string para data
            data = pd.to_datetime(data, format='%d/%m/%Y')
            return ano_referencia - data.year
        except Exception as e:
            return None  # Retorna None se não for possível converter
    return data  # Se já for um int, retorna o valor sem alteração

# Aplicando a lógica para as duas colunas
base_completa['Idade 22'] = base_completa['Idade 22'].apply(calcular_idade, ano_referencia=2022)
base_completa['Idade'] = base_completa.apply(
    lambda row: calcular_idade(row['Idade'], 2024) if row['origem'] == 'Base2024' else calcular_idade(row['Idade'], 2023),
    axis=1
)

base_completa['Idade'] = base_completa[['Idade 22', 'Idade']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Idade 22'], inplace=True)

base_completa['Defasagem'] = base_completa[['Defas', 'Defasagem']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Defas'], inplace=True)

print(base_completa.notnull().sum().sort_values(ascending=False).to_string())

# Mapeamento das classificações semelhantes
mapeamento = {
    'Privada - Programa de apadrinhamento': 'Privada',
    'Privada *Parcerias com Bolsa 100%': 'Privada',
    'Privada - Pagamento por *Empresa Parceira': 'Privada',
    'Rede Decisão': 'Privada',
    'Escola Pública': 'Pública',
    'Escola JP II': 'Pública',
    'Bolsista Universitário *Formado (a)': 'Bolsista',
    'Nenhuma das opções acima': 'Outros',
    'Concluiu o 3º EM': 'Outros'
}

# Aplicar o mapeamento para a coluna 'Instituição de ensino'
base_completa['Instituição de ensino'] = base_completa['Instituição de ensino'].replace(mapeamento)

# Verificar as classificações únicas após o agrupamento
base_completa['Instituição de ensino'].unique()

# Mapeamento das classificações semelhantes
mapeamento_genero = {
    'Menina': 'Feminino',
    'Menino': 'Masculino'
}

# Aplicar o mapeamento para a coluna 'Gênero'
base_completa['Gênero'] = base_completa['Gênero'].replace(mapeamento_genero)

# Verificar as classificações únicas após o agrupamento
base_completa['Gênero'].unique()

import re

# Função para extrair apenas números da coluna
def extrair_numero(fase):
    # Verifica se a fase é "ALFA" e retorna 0
    if str(fase).upper() == 'ALFA':
        return 0
    # Verifica se a fase é um texto como 'FASE X' e retorna o número
    match = re.search(r'\d+', str(fase))
    if match:
        return int(match.group())
    return fase  # Caso não tenha número, retorna o valor original

# Aplicando a função na coluna 'Fase'
base_completa['Fase'] = base_completa['Fase'].apply(extrair_numero)

# Verificando os valores únicos na coluna após a transformação
print(base_completa['Fase'].unique())

base_completa.info()

# Substituindo os valores de percentual para o formato desejado
base_completa['IPS'] = base_completa['IPS'].replace({r'%': '', r',': '.'}, regex=True).astype(float) / 100

# Para garantir que o valor decimal tenha vírgula
base_completa['IPS'] = base_completa['IPS'].apply(lambda x: str(x).replace('.', ','))

# Colunas que você deseja converter para float
colunas_para_float = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'IPP', 'INDE']  # Substitua pelos nomes das suas colunas

# Substituir as vírgulas por pontos e converter para float
for coluna in colunas_para_float:
    base_completa[coluna] = base_completa[coluna].replace({',': '.'}, regex=True).astype(float)

# Verificando o tipo das colunas após a conversão
print(base_completa[colunas_para_float].dtypes)

base_completa = base_completa[base_completa['Status_entrada'] != 'Novato']

base_completa['Status_entrada'] = base_completa['Status_entrada'].replace({'Desistente': 1, 'Veterano': 0})

base_completa.drop(columns=['Pedra'], inplace = True)
base_completa.dropna(inplace=True)

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


############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)
st.subheader ("""
**Faculdade de Informática e Administração Paulista – FIAP**

**Grupo 47: Cristiano de Araujo Santos Filho, Eduardo Vilela Silva, Guilherme de Faria
Rodrigues, Marcelo Pereira Varesi e Vitória Pinheiro de Mattos**
""")
st.header("**Introdução**")

st.write("""
A INSERIR
""")

st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> DATATHON - Modelo Preditivo </h1>", unsafe_allow_html = True)

st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')

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
     input_ips = float(st.slider('Selecione a sua nota', 0, 1, key='nota_ips'))

# Cria a segunda linha com 1 coluna para a quarta coluna
col4, col5, col6, col7 = st.columns(4) # Cria a quarta coluna abaixo da primeira

# Coloca conteúdo na quarta coluna
with col4:
    st.write('### IDA')
    input_ida = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ida'))

with col5:
    st.write('### IPV')
    input_ipv = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ipv'))

with col6:
    st.write('### IAN')
    input_ian = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ian'))

with col7:
    st.write('### IPP')
    input_ipp = float(st.slider('Selecione a sua nota', 0, 10, key='nota_ipp'))

st.markdown("<hr style='border:1px solid #FFFFFF;'>", unsafe_allow_html=True)

# Gênero
st.write('### Género')
input_genero = st.selectbox('Qual é o seu género?', base_completa['Gênero'].unique())

# Idade
st.write('### Idade')
input_idade = st.selectbox('Qual é o seu género?', base_completa['Idade'].unique())

# Instituição de Ensino
st.write('### Instituição de Ensino')
input_ensino = st.selectbox('Qual é a sua instituição de ensino?', base_completa['Instituição de ensino'].unique())
