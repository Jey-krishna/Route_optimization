import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from dash.dependencies import Input, Output
import plotly.graph_objs as go

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title= "Airways conditions viz",
    page_icon= "📊",
    layout= 'wide'
)

st.markdown("<h1 style='text-align: center'>Flight Route Optimization</h1>", unsafe_allow_html=True)

sns.set_style('dark')
sns.set(rc={
    'axes.facecolor' :"#27a102",
    'figure.facecolor' : "#000000"
})
df = pd.read_csv('shipment_data.csv')

nav = st.sidebar.radio('Contents', ['Explore the data', 'Prediction', "Route Optimization"])
a = 100

dfs = df.drop(columns=['Aircraft_ICAO24','Callsign', 'Origin_Country', 'Squawk_Code', 'Time_Position'])

LR1 = LinearRegression()
x1 = dfs.drop(columns='Barometric_Altitude')
y1 = dfs['Barometric_Altitude']
LR1.fit(x1,y1)

LR2 = LinearRegression()
x2 = dfs.drop(columns='Geometric_Altitude')
y2 = dfs['Geometric_Altitude']
LR2.fit(x2,y2)

LR3 = LinearRegression()
x3 = dfs.drop(columns='Velocity')
y3 = dfs['Velocity']
LR3.fit(x3,y3)

prediction1 = 0.0
prediction2 = 0.0
prediction3 = 0.0


if nav == 'Explore the data':

    st.markdown("<h2 style='text-align: center'>Explore the data</h2>", unsafe_allow_html=True)
    Average_Velocity = df["Velocity"].mean()
    Average_Barometric_Altitude= df['Barometric_Altitude'].mean()
    Average_Geoometric_Altitude = df['Geometric_Altitude'].mean()

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Average Velocity")
        st.subheader(f"{Average_Velocity:.2f}")

    with middle_column:
        st.subheader("Average Barometric Altitude")
        st.subheader(f"{Average_Barometric_Altitude:.2f}")

    with right_column:
        st.subheader("Average Geoometric Altitude")
        st.subheader(f"{Average_Geoometric_Altitude:.2f}")

    st.markdown("""---""")    

    st.markdown("<h2 style='text-align: center'>Visualization</h2>", unsafe_allow_html=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))


    axs[0].scatter(df['Velocity'], df['Geometric_Altitude'], c='#ffbf00')
    axs[0].set_xlabel('Velocity', color='white')
    axs[0].set_ylabel('Geometric Altitude', color='white')
    axs[0].tick_params(axis='x', colors='white')
    axs[0].tick_params(axis='y', colors='white')

    axs[1].scatter(df['Velocity'], df['Barometric_Altitude'], c='#ffbf00')
    axs[1].set_xlabel('Velocity', color='white')
    axs[1].set_ylabel('Barometric Altitude', color='white')
    axs[1].tick_params(axis='x', colors='white')
    axs[1].tick_params(axis='y', colors='white')

    axs[2].scatter(df['Barometric_Altitude'], df['Geometric_Altitude'], c='#ffbf00')
    axs[2].set_xlabel('Barometric Altitude', color='white')
    axs[2].set_ylabel('Geometric Altitude', color='white')
    axs[2].tick_params(axis='x', colors='white')
    axs[2].tick_params(axis='y', colors='white')

    st.pyplot(fig )
    plt.show()

    st.markdown("""---""")    

    df1, df2, df3 = np.split(df, [int(.2*len(df)), int((.5*len(df)))])

    df_filtered = df2[(df2['Longitude'].notnull()) & (df2['Latitude'].notnull())]

    fig = px.scatter_geo(df_filtered,
                        lon='Longitude',
                        lat='Latitude',
                        hover_name='Callsign',  
                        title='Aircraft Positions on World Map')

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True, countrycolor="Black",
        showcoastlines=True, coastlinecolor="Gray",
        showland=True, landcolor="rgb(217, 217, 217)",
        showocean=True, oceancolor="LightBlue"
    )

    fig.update_layout(
        height=500,
        margin={"r":15,"t":30,"l":15,"b":0}
    )

    st.plotly_chart(fig)

elif nav == 'Prediction':
    st.markdown("<h2 style='text-align: center'>Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the flight details")
    Longitude = st.number_input("longitude")
    Latitude= st.number_input("Altitude")
    Barometric_Altitude = st.number_input("Barometric_Altitude")
    Velocity=st.number_input("Velocity")
    Heading= st.number_input("Heading")
    Vertical_Rate = st.number_input("Vertical Rate")
    Geometric_Altitude= st.number_input("Barometric Altitude")

    input_data = pd.DataFrame({
        'Longitude' : [Longitude],
        'Latitude' : [Latitude],
        'Barometric_Altitude' : [Barometric_Altitude],
        'Velocity' : [Velocity],
        'Heading' : [Heading],
        'Vertical_Rate' : [Vertical_Rate],
        'Geometric_Altitude' : [Geometric_Altitude]
    })

    input_data1 = input_data.drop(columns='Barometric_Altitude')
    input_data2 = input_data.drop(columns='Geometric_Altitude')
    input_data3 = input_data.drop(columns='Velocity')

    prediction1 = LR1.predict(input_data1)
    prediction2 = LR2.predict(input_data2)
    prediction3 = LR3.predict(input_data3)
    if st.button('Predict Barometric Altitude'):
        st.write(prediction1)
    if st.button('Predict Geometric Altitude'):
        st.write(prediction2)
    if st.button('Velocity'):
        st.write(prediction3)

elif nav == 'Route Optimization':
    st.markdown("<h2 style='text-align: center'>Route Optimization</h2>", unsafe_allow_html=True)
        

    



