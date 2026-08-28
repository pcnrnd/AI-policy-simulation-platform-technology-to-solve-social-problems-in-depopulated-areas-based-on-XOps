# streamlit run chart_type.py --server.port 8506
import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb

def main():
    st.set_page_config(layout='wide')
    st.title('Data Visualization')
    
    # empty_house = pd.read_csv('./data/namwon/전라북도 남원시_빈집_20230824.csv', encoding='cp949')
    # population = pd.read_csv('./data/namwon/전라북도 남원시_인구현황_20230801.csv', encoding='cp949')
    # library = pd.read_csv('./data/namwon/전라북도 남원시_작은 도서관 현황_20231106.csv', encoding='cp949')
    # welfare_facilities = pd.read_csv('./data/namwon/welfare_facilities.csv')


    # options = st.multiselect(
    #     'Select data to check data visualization',
    #     ['empty_house', 'population', 'welfare_facilities']
    # ) 

    def reset_all_checkboxes():
        st.session_state.empty_house_check = False
        st.session_state.population_check = False
        st.session_state.welfare_facilities_check = False
        st.session_state.farm_check = False
        st.session_state.solar_power_plant_check = False
        # st.session_state.hospital_check = False


    if 'empty_house_check' not in st.session_state:
        st.session_state.empty_house_check = False
    if 'population_check' not in st.session_state:
        st.session_state.population_check = False
    if 'welfare_facilities_check' not in st.session_state:
        st.session_state.welfare_facilities_check = False
    if 'farm_check' not in st.session_state:
        st.session_state.farm_check = False
    if 'solar_power_plant_check' not in st.session_state:
        st.session_state.solar_power_plant_check = False
    # if 'hospital_check' not in st.session_state:
    #     st.session_state.hospital_check = False


    with st.container(border=True):
        st.write('Select data to check data visualization',)
        col1, col2 = st.columns(2)
        with col1:
            empty_house_check = st.checkbox("empty_house", value=st.session_state.empty_house_check)
            population_check = st.checkbox("population", value=st.session_state.population_check)
            welfare_facilities_check = st.checkbox("welfare_facilities", value=st.session_state.welfare_facilities_check)
        with col2:
            farm_check = st.checkbox('farm', value=st.session_state.farm_check)
            solar_power_plant_check = st.checkbox('solar_power_plant', value=st.session_state.solar_power_plant_check)
            # hospital_check = st.checkbox('hospital', value=st.session_state.hospital_check)
    
    st.session_state.empty_house_check = empty_house_check
    st.session_state.population_check = population_check
    st.session_state.welfare_facilities_check = welfare_facilities_check
    st.session_state.farm_check = farm_check
    st.session_state.solar_power_plant_check = solar_power_plant_check
    # st.session_state.hospital_check = hospital_check

    if st.button("초기화", use_container_width=True):
        reset_all_checkboxes()
        st.query_params.clear()
        del st.session_state.empty_house_check
        del st.session_state.population_check
        del st.session_state.welfare_facilities_check
        del st.session_state.farm_check
        del st.session_state.solar_power_plant_check
        # del st.session_state.hospital_check
        for key in st.session_state.keys():
            del st.session_state[key]

    # selected_categories = [
    #     category for category, checked in zip(
    #         ['empty_house', 'population', 'welfare_facilities', 'farm', 'solar_power_plant', 'hospital'],
    #         [empty_house_check, population_check, welfare_facilities_check, farm_check, solar_power_plant_check] # hospital_check
    #     ) if checked
    # ]        

    # if len(selected_categories) > 3:
    #     st.warning('데이터 선택은 최대 3개 까지 가능합니다.', icon="⚠️")

    if st.button('데이터 시각화 실행', use_container_width=True):
        # if len(selected_categories) <= 3:
        conn = duckdb.connect('./data/database.db') # 데이터베이스 연결

        empty_house = conn.execute('SELECT * FROM empty_house').df()
        population = conn.execute('SELECT * FROM population').df()
        welfare_facilities = conn.execute('SELECT * FROM welfare_facilities').df()

        farm = conn.execute('SELECT * FROM farm').df()
        solar_power_plant = conn.execute('SELECT * FROM solar_power_plant').df()
        # hospital = conn.execute('SELECT * FROM hospital').df()

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

        # 빈집 데이터
        if empty_house_check:    
            empty_house[['시도', '시군구', '읍면동', '도로명', '번지']] = empty_house['소재지 도로명주소'].apply(lambda x: pd.Series(split_address(x)))
            empty_house['num'] = 1  

            # 읍면동별 집계
            empty_house_sum = empty_house.groupby('읍면동')['num'].sum().reset_index()
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
        
        if population_check:
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

        if welfare_facilities_check: 
            # 복지시설 데이터
            welfare_facilities_vis = welfare_facilities.groupby('읍면동')['num'].sum().reset_index()
            col1, col2 = st.columns(2)
            with st.container():
                with col1:
                    st.subheader('남원시 복지시설 데이터')
                    st.dataframe(welfare_facilities.loc[:, ['소재지', '시설명', '읍면동', '구분']], use_container_width=True)
                    
                with col2:
                    st.subheader('남원시 복지시설 데이터 시각화')
            
                    fig = px.bar(welfare_facilities_vis, x='읍면동', y='num')
                    fig.update_xaxes(title='읍면동')
                    fig.update_yaxes(title='복지시설 수')
                    fig.update_layout(
                        xaxis_tickangle=90
                    )
                    st.plotly_chart(fig)
        # 농장
        if farm_check:
            farm_vis = farm.groupby('읍면동')['num'].sum().reset_index()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('남원시 농장 데이터')
                st.dataframe(farm, use_container_width=True)
            with col2:
                st.subheader('남원시 농장 데이터 시각화')
                fig = px.bar(farm_vis, x='읍면동', y='num')
                fig.update_xaxes(title='읍면동')
                fig.update_yaxes(title='농장 수')
                fig.update_layout(
                    xaxis_tickangle=90
                )
                st.plotly_chart(fig)


        if solar_power_plant_check:
            solar_power_plant_vis = solar_power_plant.groupby('읍면동')['num'].sum().reset_index()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('남원시 태양광 발전소 데이터')
                st.dataframe(solar_power_plant, use_container_width=True)
            with col2:
                st.subheader('남원시 태양광 발전소 데이터 시각화')
                fig = px.bar(solar_power_plant_vis, x='읍면동', y='num')
                fig.update_xaxes(title='읍면동')
                fig.update_yaxes(title='태양광 발전소 수')
                fig.update_layout(
                    xaxis_tickangle=90
                )
                st.plotly_chart(fig)

        # if hospital_check:
        #     hospital_vis = hospital.groupby('읍면동')['num'].sum().reset_index() #.drop('index', axis=1)
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.subheader('남원시 병원 데이터')
        #         st.dataframe(hospital, use_container_width=True)
            
        #     with col2:
        #         st.subheader('남원시 병원 데이터 시각화')
        #         fig = px.bar(hospital_vis, x='읍면동', y='num')
        #         fig.update_xaxes(title='읍면동')
        #         fig.update_yaxes(title='병원 수')
        #         fig.update_layout(
        #             xaxis_tickangle=90
        #         )
        #         st.plotly_chart(fig)
    

if __name__ == '__main__':
    main()