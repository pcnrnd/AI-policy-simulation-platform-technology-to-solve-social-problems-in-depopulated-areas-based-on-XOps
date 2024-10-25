# streamlit run chart_type.py --server.port 8506
import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
from lib.exception_handling import ExceptionHandling

def main():
    st.set_page_config(layout='wide')
    st.title('데이터 시각화 처리 응답속도')

    options = st.multiselect(
            "Select data for visualization",
            ["restaurant_2024", "restaurant_2023", "restaurant_2022", "restaurant_2021", "restaurant_2020"])

if __name__ == '__main__':
    main()