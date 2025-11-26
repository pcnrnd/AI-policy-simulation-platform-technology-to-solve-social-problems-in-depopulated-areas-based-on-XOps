# streamlit run old_vis_types.py --server.port 8501
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.set_page_config(layout='wide')
    st.title('Data Visualization')
    
    PATH = './data/east_side_travel_data/tn_adv_consume_his_사전소비내역_F.csv'
    consumption_df = pd.read_csv(PATH)

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('지역별 결제 금액 데이터')
            st.dataframe(consumption_df.loc[:, ('ROAD_NM_ADDR', 'PAYMENT_AMT_WON')], use_container_width=True)

        with col2:
            st.subheader('지역별 결제 금액 시각화')
            fig = px.bar(consumption_df, x='ROAD_NM_ADDR', y='PAYMENT_AMT_WON')
            fig.update_xaxes(title='지역')
            fig.update_yaxes(title='결제 금액')
            st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('지역별 결제 건수 데이터')
            st.dataframe(consumption_df.loc[:, ('ROAD_NM_ADDR', 'PAYMENT_NUM')], use_container_width=True)

        with col2:
            st.subheader('지역별 결제 건수 시각화')
            fig = px.bar(consumption_df, x='STORE_NM', y='PAYMENT_NUM')
            fig.update_xaxes(title='지역')
            fig.update_yaxes(title='결제 건수')
            st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('시설별 결제 금액 데이터')
            st.dataframe(consumption_df.loc[:, ('STORE_NM', 'PAYMENT_AMT_WON')], use_container_width=True)

        with col2:
            st.subheader('시설별 결제 금액 시각화')
            fig = px.bar(consumption_df, x='STORE_NM', y='PAYMENT_AMT_WON')
            fig.update_xaxes(title='시설')
            fig.update_yaxes(title='결제 금액')
            st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('시설별 결제 건수 데이터')
            st.dataframe(consumption_df.loc[:, ('STORE_NM', 'PAYMENT_NUM')], use_container_width=True)

        with col2:
            st.subheader('시설별 결제 건수 시각화')
            fig = px.bar(consumption_df, x='STORE_NM', y='PAYMENT_NUM')
            fig.update_xaxes(title='시설')
            fig.update_yaxes(title='결제 건수')
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()