# streamlit run vis_types.py --server.port 8501
import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.set_page_config(layout='wide')
    st.title('Data Visualization')
    
    # empty_house = pd.read_csv('./data/namwon/전라북도 남원시_빈집_20230824.csv', encoding='cp949')
    # population = pd.read_csv('./data/namwon/전라북도 남원시_인구현황_20230801.csv', encoding='cp949')
    # library = pd.read_csv('./data/namwon/전라북도 남원시_작은 도서관 현황_20231106.csv', encoding='cp949')
    # welfare_facilities = pd.read_csv('./data/namwon/welfare_facilities.csv')
    
    conn = duckdb.connect('./data/database.db')

    empty_house = conn.execute('SELECT * FROM empty_house').df()
    population = conn.execute('SELECT * FROM population').df()
    library = conn.execute('SELECT * FROM library').df()
    welfare_facilities = conn.execute('SELECT * FROM welfare_facilities').df()

    def split_address(address):
        parts = address.split()
        
        if len(parts) >= 4:
            city_name = parts[0]  # 예: 전라북도
            district = parts[1]  # 예: 남원시
            town = parts[2]  # 예: 대강면
            road_name = ' '.join(parts[3:-1])  # 예: 대강월산길 (주소의 마지막 요소 전까지 모두 도로명으로 처리)
            bungee = parts[-1]  # 예: 37-16 (주소의 마지막 요소는 항상 번지로 처리)
            
            return city_name, district, town, road_name, bungee
        else:
            return None, None, None, None, None
        
    empty_house[['시도', '시군구', '읍면동', '도로명', '번지']] = empty_house['소재지 도로명주소'].apply(lambda x: pd.Series(split_address(x)))
    empty_house['num'] = 1  

    # 읍면동별 집계
    empty_house_sum = empty_house.groupby('읍면동')['num'].sum().reset_index()

    library[['시도', '시군구', '읍면동', '도로명', '번지']] = library['위치'].apply(lambda x: pd.Series(split_address(x)))
    library['num'] = 1  
    
    library_sum = library[['도서관명', '시도', '시군구', '읍면동', '도로명', '번지', 'num']]

    # 빈집 데이터
    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('남원시 빈집 데이터')
            st.dataframe(empty_house.loc[:, ['시도', '시군구', '읍면동', '도로명', '번지']], use_container_width=True)

        with col2:
            st.subheader('남원시 빈집 데이터 시각화')
            fig = px.bar(empty_house_sum, x='읍면동', y='num')
            fig.update_xaxes(title='읍면동')
            fig.update_yaxes(title='빈집 수')
            fig.update_layout(
                xaxis_tickangle=90
            )
            st.plotly_chart(fig)
    
    # 인구 데이터
    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('남원시 인구 데이터')
            st.dataframe(population.loc[:, ['연도별', '월별', '읍면동', '인구수', '인구수(남)', 
                                            '인구수(여)', '세대수', '60세 이상 인구수',
                                            '60세 이상 인구수(남)', '60세 이상 인구수(여)']], use_container_width=True)

        with col2:
            st.subheader('남원시 인구 데이터 시각화')
            fig = px.bar(population, x='읍면동', y=['60세 이상 인구수(남)', '60세 이상 인구수(여)', '인구수(남)', '인구수(여)'])
            fig.update_xaxes(title='읍면동')
            fig.update_yaxes(title='인구 수')
            fig.update_layout(
                xaxis_tickangle=90
            )
            st.plotly_chart(fig)

    # 복지시설 데이터
    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            st.subheader('남원시 복지시설 데이터')
            st.dataframe(welfare_facilities.loc[:, ['소재지', '시설명', '읍면동', '구분']], use_container_width=True)
            
        with col2:
            st.subheader('남원시 복지시설 데이터 시각화')
    
            fig = px.bar(welfare_facilities, x='읍면동', y='num')
            fig.update_xaxes(title='읍면동')
            fig.update_yaxes(title='복지시설 수')
            fig.update_layout(
                xaxis_tickangle=90
            )
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()