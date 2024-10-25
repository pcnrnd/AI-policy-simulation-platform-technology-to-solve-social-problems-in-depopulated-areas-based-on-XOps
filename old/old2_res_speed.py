# streamlit run res_speed.py --server.port 8502
import time
import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from lib.exception_handling import ExceptionHandling

def main():
    st.set_page_config(layout='wide')
    st.title('Data Visualization')

    # PATH = './data/speed_data.csv'
    # df = pd.read_csv(PATH)

    con = duckdb.connect('./data/database.db')
    df = con.execute("SELECT * FROM speed_data").df()

    ex = ExceptionHandling()

    start_time = datetime.now()
    background_color = '#F0F8FF' # AliceBlue
    text_color = '#708090' # SlateGray
    placeholder = st.empty()  # 초기 메시지와 업데이트할 공간을 생성
    # 초기 메시지를 표시
    initial_message = f"""
    <div style="padding: 10px; background-color: {background_color}; border-radius: 10px; text-align: center;">
        <h3 style="color: {text_color};">Calculating response time...</h3>
    </div>
    """
    placeholder.markdown(initial_message, unsafe_allow_html=True)


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
            st.plotly_chart(fig)
            ex.exception_check()

        with col3:
            st.subheader('시도별 음식점 종류')
            category_df = df.groupby(['시도', '구분'])['num'].agg('sum').reset_index()
            fig = px.scatter(category_df, x='시도', y='num', color='구분')
            fig.update_layout(
                xaxis_tickangle=90
            )
            st.plotly_chart(fig)
            ex.exception_check()

    col1, col2 = st.columns(2)
    with st.container(): 
        with col1:
            st.subheader('시도별 음식점 수 변화')
            city_df = pd.DataFrame(df.groupby(['시도', '인허가일자'])['num'].agg('sum')).reset_index()
            fig = px.area(city_df, x='인허가일자', y='num', color='시도')
            st.plotly_chart(fig)
            ex.exception_check()

        with col2:
            st.subheader('음식점 종류 변화')
            city_df = pd.DataFrame(df.groupby(['구분', '인허가일자'])['num'].agg('sum')).reset_index()
            fig = px.line(city_df, x='인허가일자', y='num', color='구분')
            st.plotly_chart(fig)
            ex.exception_check()

    with st.container(): 
        st.dataframe(df.iloc[:, :-1], use_container_width=True)

    time.sleep(1)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # background_color = '#F0F8FF' # AliceBlue
    # text_color = '#708090' # SlateGray

    # 최종 응답 시간으로 메시지를 업데이트합니다
    final_message = f"""
    <div style="padding: 10px; background-color: {background_color}; border-radius: 10px; text-align: center;">
        <h3 style="color: {text_color};">Response time: {elapsed_time:.6f} seconds</h3>
    </div>
    """
    placeholder.markdown(final_message, unsafe_allow_html=True)

if __name__ == '__main__':
    main()