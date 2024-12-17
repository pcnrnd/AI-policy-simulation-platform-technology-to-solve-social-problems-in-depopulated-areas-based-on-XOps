import pandas as pd
import streamlit as st
import plotly.express as px


class Visualization:
    def __init__(self):
        pass

    def bar_chart(self, df, group):
        selected_df = pd.DataFrame(df.groupby(group)['num'].agg('sum')).reset_index()
        fig = px.bar(selected_df, x=group, y='num', color=group)
        fig.update_layout(
            xaxis_tickangle=90
        )
        st.plotly_chart(fig)

    
