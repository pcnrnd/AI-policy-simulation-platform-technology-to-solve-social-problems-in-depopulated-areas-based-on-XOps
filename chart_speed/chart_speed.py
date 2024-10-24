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
    
if __name__ == '__main__':
    main()