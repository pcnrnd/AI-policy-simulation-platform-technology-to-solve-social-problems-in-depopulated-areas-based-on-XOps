# streamlit run chart_type.py --server.port 8506
import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
from lib.exception_handling import ExceptionHandling
from lib.visualization import Visualization

def main():
    st.set_page_config(layout='wide')
    st.title('데이터 시각화 처리 응답속도')

    options = st.multiselect(
            "Select data for visualization",
            ["restaurant_2024", "restaurant_2023", "restaurant_2022", "restaurant_2021", "restaurant_2020"])

    if st.button('데이터 시각화 실행', use_container_width=True):    
        # 빈집
        try:
            ex = ExceptionHandling()
            vis = Visualization


            conn = duckdb.connect('./data/restaurant/database.db')
            df = conn.execute(f'SELECT * FROM {options[0]}').df()

            col1, col2, col3 = st.columns(3)

            with st.container():
                with col1:
                    # restaurant_2024
                    st.subheader('시도별 음식점 수')
                    # selected_df = pd.DataFrame(df.groupby('시도')['num'].agg('sum')).reset_index()
                    # fig = px.bar(selected_df, x='시도', y='num', color='시도')
                    # fig.update_layout(
                    #     xaxis_tickangle=90
                    # )
                    # st.plotly_chart(fig)
                    vis.bar_chart(df, group='시도')
                    ex.exception_check()

                with col2:
                    # restaurant_2023
                    st.subheader('시도별 영업 상태')
                    # selected_df = pd.DataFrame(df.groupby(['시도', '영업상태명'])['num'].agg('sum')).reset_index()
                    # fig = px.funnel(selected_df, x='num', y='시도', color='영업상태명')
                    # # fig = px.bar(selected_df, x='시도', y='num', color='영업상태명')
                    # st.plotly_chart(fig)
                    vis.funnel_chart(df, group_a='시도', group_b='영업상태명')
                    ex.exception_check()

                with col3:
                    # restaurant_2022
                    st.subheader('시도별 음식점 종류')
                    category_df = df.groupby(['시도', '구분'])['num'].agg('sum').reset_index()
                    fig = px.scatter(category_df, x='시도', y='num', color='구분')
                    fig.update_layout(
                        xaxis_tickangle=90
                    )
                    st.plotly_chart(fig)
                    ex.exception_check()

            conn.close()
        except IndexError:
            st.write('데이터를 선택해 주세요')

if __name__ == '__main__':
    main()