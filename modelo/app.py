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

# Ocupação
st.write('### Género')
input_genero = st.selectbox('Qual é o seu género?', base_completa['Gênero'].unique())

# Lista de todas as variáveis: 
novo_cliente = [0, # ID_Cliente
                    input_carro_proprio, # Tem_carro
                    input_casa_propria, # Tem_Casa_Propria
                    telefone_trabalho, # Tem_telefone_trabalho
                    telefone, # Tem_telefone_fixo
                    email,  # Tem_email
                    membros_familia,  # Tamanho_Familia
                    input_rendimentos, # Rendimento_anual	
                    input_idade, # Idade
                    input_tempo_experiencia, # Anos_empregado
                    input_categoria_renda, # Categoria_de_renda
                    input_grau_escolaridade, # Grau_Escolaridade
                    input_estado_civil, # Estado_Civil	
                    input_tipo_moradia, # Moradia                                                  
                    input_ocupacao, # Ocupacao
                     0 # target (Mau)
                    ]


# Separando os dados em treino e teste
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)

#Criando novo cliente
cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)

#Concatenando novo cliente ao dataframe dos dados de teste
teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)

#Pipeline
def pipeline_teste(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMaxWithFeatNames()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

#Aplicando a pipeline
teste_novo_cliente = pipeline_teste(teste_novo_cliente)

#retirando a coluna target
cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

#Predições 
if st.button('Enviar'):
    model = joblib.load('modelo/xgb.joblib')
    final_pred = model.predict(cliente_pred)
    if final_pred[-1] == 0:
        st.success('### Parabéns! Você teve o cartão de crédito aprovado')
        st.balloons()
    else:
        st.error('### Infelizmente, não podemos liberar crédito para você agora!')
 
