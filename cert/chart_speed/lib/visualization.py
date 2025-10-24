import pandas as pd
import streamlit as st
import plotly.express as px


class Visualization:
    def __init__(self):
        pass

    def bar_chart(self, df, group):
        '''
        df: dataframe \n
        group: column name
        '''
        selected_df = pd.DataFrame(df.groupby(group)['num'].agg('sum')).reset_index()
        fig = px.bar(selected_df, x=group, y='num', color=group)
        fig.update_layout(
            xaxis_tickangle=90
        )
        st.plotly_chart(fig)

    def funnel_chart(self, df, group_a, group_b):
        '''
        df: dataframe \n
        group_a: column name a
        group_b: column name b
        '''
        selected_df = pd.DataFrame(df.groupby([group_a, group_b])['num'].agg('sum')).reset_index()
        fig = px.funnel(selected_df, x='num', y=group_a, color=group_b)
        st.plotly_chart(fig)

    def scatter_chart(slef, df, group_a, group_b):
        '''
        df: dataframe \n
        group_a: column name a \n
        group_b: column name b
        '''
        category_df = df.groupby([group_a, group_b])['num'].agg('sum').reset_index()
        fig = px.scatter(category_df, x=group_a, y='num', color=group_b)
        fig.update_layout(
            xaxis_tickangle=90
        )
        st.plotly_chart(fig)
        

    def line_chart(self, df, group_a, group_b):
        category_df = df.groupby([group_a, group_b])['num'].agg('sum').reset_index()
        fig = px.line(category_df, x=group_a, y='num', color=group_b)
        fig.update_layout(
            xaxis_tickangle=90
        )
        st.plotly_chart(fig)

    def pie_chart_by_category(self, df, group_by='시도'):
        """
        전체 지역별 업종 분포 파이차트
        """
        # NaN 값 제거
        df_copy = df.dropna(subset=[group_by, '구분'])
        
        # 전체 지역별 업종 분포 집계
        category_df = df_copy.groupby([group_by, '구분'])['num'].sum().reset_index()
        
        # 데이터가 있는지 확인
        if len(category_df) == 0:
            st.warning("표시할 데이터가 없습니다.")
            return
        
        # 시각화 - 전체 지역별 업종 분포
        fig = px.pie(category_df, values='num', names='구분', 
                     title=f'전체 업종 분포',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def trend_chart_by_category(self, df, group_by='구분'):
        """
        업종별 인허가일자 추세 분석
        """
        # 인허가일자를 날짜 형식으로 변환
        df_copy = df.copy()
        df_copy['인허가일자'] = pd.to_datetime(df_copy['인허가일자'], errors='coerce')
        
        # 날짜별로 그룹화하여 월별 집계
        df_copy['년월'] = df_copy['인허가일자'].dt.to_period('M')
        
        # 업종별, 년월별 집계
        trend_df = df_copy.groupby([group_by, '년월'])['num'].sum().reset_index()
        trend_df['년월'] = trend_df['년월'].astype(str)
        
        # 시각화
        fig = px.line(trend_df, x='년월', y='num', color=group_by,
                     title=f'{group_by}별 인허가일자 추세',
                     labels={'num': '음식점 수', '년월': '인허가 년월'})
        
        fig.update_layout(
            xaxis_tickangle=45,
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)