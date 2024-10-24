import requests
import duckdb
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from lib.exception_handling import ExceptionHandling

def main():
    st.set_page_config(layout='wide')
    st.title('데이터 시각화 처리 응답속도')

    twenty_zero = 'restaurant_2020' # 22년 데이터
    twenty_one = 'restaurant_2021' # 23년 데이터
    twenty_two = 'restaurant_2022' # 22년 데이터
    twenty_three = 'restaurant_2023' # 23년 데이터
    twenty_four = 'restaurant_2024' # 24년 데이터

    options = st.multiselect(
        'Select data to check response speed',
        [twenty_four, twenty_three, twenty_two, twenty_one, twenty_zero]
    ) 
    
    try:
        if st.button('데이터 시각화 실행', use_container_width=True): # options:
            # PATH = './data/speed_data.csv'
            # df = pd.read_csv(PATH)
            conn = duckdb.connect('./data/database.db')
            df = conn.execute(f"SELECT * FROM {options[0]}").df()
            ex = ExceptionHandling()
            
            col1, col2, col3 = st.columns(3)
            with st.container():
                with col1:
                    st.subheader('시도별 음식점 수')
                    selected_df = pd.DataFrame(df.groupby('시도')['num'].agg('sum')).reset_index()
                    fig = px.bar(selected_df, x='시도', y='num', color='시도')
                    fig.update_layout(
                        xaxis_tickangle=90
                    )
                    st.plotly_chart(fig)
                    ex.exception_check()
                
                with col2:
                    st.subheader('시도별 영업 상태')
                    selected_df = pd.DataFrame(df.groupby(['시도', '영업상태명'])['num'].agg('sum')).reset_index()
                    fig = px.funnel(selected_df, x='num', y='시도', color='영업상태명')
                    # fig = px.bar(selected_df, x='시도', y='num', color='영업상태명')
                    st.plotly_chart(fig)
                    ex.exception_check()

    except IndexError:
        st.write('데이터를 선택하세요')
if __name__ == '__main__':
    main()