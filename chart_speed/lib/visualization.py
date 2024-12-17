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