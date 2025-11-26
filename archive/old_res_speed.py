# streamlit run old_res_speed.py --server.port 8503
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
def main():
    st.set_page_config(layout='wide')
    st.title('Data Visualization')
    
    PATH = './data/filtered_data.csv'
    df = pd.read_csv(PATH)

    start_time = datetime.now()

    background_color = '#F0F8FF' # AliceBlue
    text_color = '#708090' # SlateGray
    
    placeholder = st.empty()  # 초기 메시지와 나중에 업데이트할 공간을 생성합니다

    # 초기 메시지를 표시합니다
    initial_message = f"""
    <div style="padding: 10px; background-color: {background_color}; border-radius: 10px; text-align: center;">
        <h3 style="color: {text_color};">Calculating response time...</h3>
    </div>
    """
    placeholder.markdown(initial_message, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)
    with st.container():
        with col1:
            fig = px.scatter(df, x=df.columns[3], y=df.columns[2], color=df.columns[-2])
            st.plotly_chart(fig)
            time.sleep(1)
        with col2:
            fig = px.line(df.iloc[:, [2, 3]])
            st.plotly_chart(fig)
            time.sleep(1)
        with col3:
            fig = px.bar(df, x=df.columns[-2], y=df.columns[3], color=df.columns[-2])
            st.plotly_chart(fig)
            time.sleep(1)

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            time_df = pd.read_csv('./data/time_data.csv')
            max_df = time_df[time_df['X_ActualVelocity'] == time_df.iloc[:, :-3].max().values[1]]
            fig = px.timeline(max_df, x_start="start", x_end="end", color='Machining_Process') # , color=test.columns[-3]
            st.plotly_chart(fig)   
            time.sleep(1)     

        with col2:
            fig = px.area(df, x=df.columns[1], y=df.columns[2], color=df.columns[-2])
            st.plotly_chart(fig)
            time.sleep(1)

    with st.container():    
        st.dataframe(df, use_container_width=True)


    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    background_color = '#F0F8FF' # AliceBlue
    text_color = '#708090' # SlateGray

    # 최종 응답 시간으로 메시지를 업데이트합니다
    final_message = f"""
    <div style="padding: 10px; background-color: {background_color}; border-radius: 10px; text-align: center;">
        <h3 style="color: {text_color};">Response time: {elapsed_time:.6f} seconds</h3>
    </div>
    """
    placeholder.markdown(final_message, unsafe_allow_html=True)

if __name__ == '__main__':
    main()