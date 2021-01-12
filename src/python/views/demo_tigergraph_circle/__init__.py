import asyncio
import graphistry
import os
import pandas as pd
import streamlit as st
from components import GraphistrySt, URLParam
from neptune_helper import gremlin_helper, df_helper
from css import all_css
from util import getChild
import time
import altair as alt
from TigerGraph_helper import tg_helper
import plotly.express as px

from gremlin_python import statics
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import WithOptions, T

############################################
#
#   DASHBOARD SETTINGS
#
############################################
#  Controls how entrypoint.py picks it up


app_id = 'tigergraph_circle'
logger = getChild(app_id)
urlParams = URLParam(app_id)
node_id_col = 'id'
src_id_col = 'from_id'
dst_id_col = 'to_id'
node_label_col = 'Source_Type'
edge_label_col = 'Destination_Type'

# Setup a structure to hold metrics
metrics = {'tigergraph_time': 0, 'graphistry_time': 0,
           'node_cnt': 0, 'edge_cnt': 0, 'prop_cnt': 0}


conn = tg_helper.connect_to_tigergraph()

# Define the name of the view
def info():
    return {
        'id': app_id,
        'name': 'TigerGraph: Fraud Filter circle',
        'tags': ['demo', 'tigergraph_demo_circle']
    }


def run():
    run_all()


############################################
#
#   PIPELINE PIECES
#
############################################


# Have fun!
def custom_css():
    all_css()
    st.markdown(
        """<style>

        </style>""", unsafe_allow_html=True)


# Given URL params, render left sidebar form and return combined filter settings
# https://docs.streamlit.io/en/stable/api.html#display-interactive-widgets
def sidebar_area():
    # q = conn.runInstalledQuery("mostDirectInfections")
    # Most_infectious_IDS = q[0]['Answer']
    # MI_List = [d['v_id'] for d in Most_infectious_IDS if 'v_id' in d]
    num_edges_init = urlParams.get_field('num_matches', 0.5)
    # MI_List.reverse()
    idList = [i for i in range(1, 500)]
    user_id = st.sidebar.selectbox(
        'User ID ',
        idList
    )


    urlParams.set_field('user_id', user_id)

    return { 'user_id': user_id}


def plot_url(nodes_df, edges_df):
    global metrics

    logger.info('Starting graphistry plot')
    tic = time.perf_counter()
    g = graphistry\
        .edges(edges_df)\
        .bind(source='from_id', destination='to_id') \
        .bind(edge_title='amount', edge_label='amount')\
        .nodes(nodes_df) \
        .bind(node='n') \
        .addStyle(bg={'color': 'white'})\
        .encode_point_color("color")\
        .encode_edge_color("color")\
        .encode_point_icon('type', categorical_mapping={'User': 'laptop',
                                                       'Transaction': 'server'},
                          default_mapping='question')


    # if not (node_label_col is None):
    #     g = g.bind(point_title=node_label_col)

    # if not (edge_label_col is None):
    #     g = g.bind(edge_title=edge_label_col)

    url = g.plot(render=False)
        
    toc = time.perf_counter()
    metrics['graphistry_time'] = toc-tic
    logger.info(f'Graphisty Time: {metrics["graphistry_time"]}')
    logger.info('Generated viz, got back urL: %s', url)

    return url

import pyTigerGraphBeta as tg
import pandas as pd
import datetime
#
# def flatten(lst_of_lst):
#     try:
#         if type(lst_of_lst[0]) == list:
#             return [item for sublist in lst_of_lst for item in sublist]
#         else:
#             return lst_of_lst
#     except:
#         print('fail', lst_of_lst)
#         return lst_of_lst
#
#


# Given filter settings, generate/cache/return dataframes & viz
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def run_filters( user_id):
    global metrics
    global conn

    if conn is None:
        conn = tg.TigerGraphConnection(host="https://fraud-streamlit.i.tgcloud.io", graphname="AntiFraud", username="tigergraph", password="tigergraph")
    
    conn.getToken("tufp2os5skgljafj7ol4ikht2atc7rbj")

    logger.info('Querying Tigergraph')
    tic = time.perf_counter()

    results = conn.runInstalledQuery("circleDetection", {"srcId": user_id,"stepHighLimit":7},sizeLimit=1000000000
                                     ,timeout=120000)
    results = results[0]['@@circleEdgeTuples']


    out = []
    from_ids = []
    to_ids = []
    times = []
    amounts = []
    types = []
    from_types = []
    to_types = []
    
    for o in results:
        for s in o:
            if {"from_id":s["e"]["from_id"],"to_id":s["e"]["to_id"],"amount":s["amount"],"time":s["ts"],"type":s["e"]["e_type"]} not in out:
                out.append({"from_id":s["e"]["from_id"],"to_id":s["e"]["to_id"],"amount":s["amount"],"time":s["ts"],"type":s["e"]["e_type"]})
                from_ids.append(s["e"]["from_id"])
                to_ids.append(s["e"]["to_id"])
                amounts.append(s["amount"])
                times.append(s["ts"])
                types.append(s["e"]["e_type"])
                from_types.append(s['e']['from_type'])
                to_types.append(s['e']['from_type'])

    ############# BEGIN
    edges_df = pd.DataFrame({
        'from_id': from_ids,
        'to_id': to_ids,
        'amount': amounts,
        'time': times,
        'type': types
    })
    node_idf = []
    typef = []
    for i in range(len(from_ids)):
        if from_ids[i] not in node_idf:
            node_idf.append(from_ids[i])
            typef.append(from_types[i])
        if to_ids[i] not in node_idf:
            node_idf.append(to_ids[i])
            typef.append(to_types[i])
    
    nodeType2color = {
        'User': 0x00000000,         # black
        'Transaction': 0xFF000000   # red
    }

    edgeType2color = {
        'User_Transfer_Transaction': 0x00FF0000,
        'User_Recieve_Transaction_Rev': 0x0000FF00
    }


    nodes_df = pd.DataFrame({
        'n': node_idf ,
        'type': typef,
        'size':0.1
    })
    nodes_df['color'] = nodes_df['type'].apply(lambda type_str: nodeType2color[type_str])
    edges_df['color'] = edges_df['type'].apply(lambda type_str: edgeType2color[type_str])

    ############### END

    try:

        res = nodes_df.values.tolist()
        toc = time.perf_counter()
        logger.info(f'Query Execution: {toc-tic:0.02f} seconds')
        logger.debug('Query Result Count: %s', len(res))
        metrics['tigergraph_time'] = toc-tic

        # Calculate the metrics
        metrics['node_cnt'] = nodes_df.size
        metrics['edge_cnt'] = edges_df.size
        metrics['prop_cnt'] = (nodes_df.size * nodes_df.columns.size) + \
            (edges_df.size * edges_df.columns.size)

        if nodes_df.size > 0:
            url = plot_url(nodes_df, edges_df)
        else:
            url = ""
    except Exception as e:
        logger.error('oops in TigerGraph', exc_info=True)
        raise e
    logger.info("Finished compute phase")

    try:
        pass

    except RuntimeError as e:
        if str(e) == "There is no current event loop in thread 'ScriptRunner.scriptThread'.":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        else:
            raise e

    except Exception as e:
        logger.error('oops in TigerGraph', exc_info=True)
        raise e

    return {'nodes_df': nodes_df, 'edges_df': edges_df, 'url': url, 'res': res}


def main_area(url, nodes, edges, user_id):
    global conn

    logger.debug('rendering main area, with url: %s', url)
    GraphistrySt().render_url(url)

    dates = []
    amounts = []
    transfer_type = []
    results = None
    if conn is None:
        conn = tg.TigerGraphConnection(host="https://fraud-streamlit.i.tgcloud.io", graphname="AntiFraud", username="tigergraph", password="tigergraph")
    
    conn.getToken("tufp2os5skgljafj7ol4ikht2atc7rbj")
    try:
        results = conn.runInstalledQuery("totalTransaction", params={"Source": user_id})[0]
    except Exception as e:
        print(e)
    
    # Create bar chart of transactions
    if results != None:
        for action in results:
            for transfer in results[action]:
                dates.append(datetime.datetime.fromtimestamp(transfer['attributes']['ts']))
                amounts.append(transfer['attributes']['amount'])
                transfer_type.append(action)
        cols = list(zip(dates, amounts, transfer_type))
        cols = sorted(cols, key=lambda x: x[0].day)
        cols = sorted(cols, key=lambda x: x[0].month)
        cols = sorted(cols, key=lambda x: x[0].year)
        df = pd.DataFrame(data=cols, columns=['Date', 'Amount', 'Type'])
        df['Date'] = pd.to_datetime(df['Date'])
        map_color = {"receive":"rgba(0,0,255,0.5)", "transfer":"rgba(255,0,0,0.5)"}
        df['Color'] = df['Type'].map(map_color)

        df = df.groupby([df['Date'].dt.to_period('M'), 'Type', 'Color']).sum()
        df = df.reset_index(level=['Type', 'Color'])
        df.index = df.index.values.astype('datetime64[M]')
        bar = px.bar(df, x=df.index, y='Amount', labels={'x': 'Date'}, color='Type', color_discrete_map = map_color, text='Amount', title="Transaction Amounts by Month for User {}".format(user_id), height=350, barmode='group')
        bar.update_xaxes(
            dtick="M1",
            tickformat="%b\n%Y")
        try:
            for trace in bar.data:
                trace.name = trace.name.split('=')[1].capitalize()
        except:
            for trace in bar.data:
                trace.name = trace.name.capitalize()

        st.plotly_chart(bar, use_container_width=True)

    st.markdown(f'''<small>
            TigerGraph Load Time (s): {float(metrics['tigergraph_time']):0.2f} | 
            Graphistry Load Time (s): {float(metrics['graphistry_time']):0.2f} | 
            Node Count: {metrics['node_cnt']} |  
            Edge Count: {metrics['edge_cnt']} | 
            Property Count: {metrics['prop_cnt']}  
        </small>''', unsafe_allow_html=True)


############################################
#
#   PIPELINE FLOW
#
############################################


def run_all():

    custom_css()

    try:

        # Render sidebar and get current settings
        sidebar_filters = sidebar_area()

        # Compute filter pipeline (with auto-caching based on filter setting inputs)
        # Selective mark these as URL params as well
        filter_pipeline_result = run_filters(**sidebar_filters)

        # Render main viz area based on computed filter pipeline results and sidebar settings if data is returned
        if filter_pipeline_result['nodes_df'].size > 0:
            main_area(filter_pipeline_result['url'],
                      filter_pipeline_result['nodes_df'],
                      filter_pipeline_result['edges_df'],
                      sidebar_filters['user_id'])
        else:  # render a message
            st.write("No data matching the specfiied criteria is found")

    except Exception as exn:
        st.write('Error loading dashboard')
        st.write(exn)