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


app_id = 'tigergraph_fraud' 
logger = getChild(app_id)
urlParams = URLParam(app_id)
node_id_col = 'id'
src_id_col = 'Source'
dst_id_col = 'Destination'
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
        'name': 'TigerGraph: Fraud Filter',
        'tags': ['demo', 'tigergraph_demo_fraud']
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

    max_trust = st.sidebar.slider(
        'Maximum Trust Score', min_value=0.01, max_value=1.00, value=num_edges_init)
    urlParams.set_field('max_trust', max_trust)
    urlParams.set_field('user_id', user_id)

    return {'max_trust': max_trust, 'user_id': user_id}


def plot_url(nodes_df, edges_df):
    global metrics
    # nodes_df = df_helper.flatten_df(nodes_df)
    # edges_df = df_helper.flatten_df(edges_df)

    logger.info('Starting graphistry plot')
    tic = time.perf_counter()
    g = graphistry\
        .edges(edges_df)\
        .bind(edge_color='my_color', source='Source', destination='Destination')\
        .nodes(nodes_df)\
        .bind(node=src_id_col)     

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


# def path_to_df(p):
#     nodes = {}
#     edges = {}

#     for triple in p:

#         src_id = triple[0][T.id]
#         nodes[src_id] = df_helper.vertex_to_dict(triple[0])

#         dst_id = triple[2][T.id]
#         nodes[dst_id] = df_helper.vertex_to_dict(triple[2])

#         edges[triple[1][T.id]] = df_helper.edge_to_dict(
#             triple[1], src_id, dst_id)

#     return pd.DataFrame(nodes.values()), pd.DataFrame(edges.values())

import pyTigerGraphBeta as tg
import pandas as pd
import datetime

def type_to_color(t):
    mapper = {'User': 0xFF000000}
    if t in mapper:
        return mapper[t]
    else:
        return 0xFFFFFF00

# Given filter settings, generate/cache/return dataframes & viz
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def run_filters(max_trust, user_id):
    global metrics
    global conn

    if conn is None:
        conn = tg.TigerGraphConnection(host="https://fraud-graph-kit.i.tgcloud.io", graphname="AntiFraud", username="tigergraph", password="tigergraph")
    
    conn.getToken("g8ivecpsuei7hs0hf8i0a2kcfi5oq0pl")
    
    # secret = conn.createSecret()
    #token = conn.getToken("hna88qpb3g87l3b8qcv1v01eju75jnqr", setToken=True)
    
    ##################


    logger.info('Querying Tigergraph')
    tic = time.perf_counter()

    ################
    #t = g.V().inE()
    source_col = []
    dest_col = []
    source_type = []
    dest_type = []
    # q = conn.runInstalledQuery("getGraph")
    # edges = q[0]['@@AllE']

    q = conn.runInstalledQuery("fraudConnectivity",{"inputUser":user_id, "trustScore":max_trust})
    edges = q[1]['@@visResult']

    for edge in edges:
        source_col.append(edge['from_id'])
        source_type.append(edge['from_type'])
        dest_col.append(edge['to_id'])
        dest_type.append(edge['to_type'])

    nodes_df = pd.DataFrame(list(zip(source_col, source_type, dest_col, dest_type)), columns=['Source', 'Source_Type', 'Destination', 'Destination_Type'])
    edges_df = nodes_df.assign(my_color=nodes_df['Source_Type'].apply(lambda v: type_to_color(v)))
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
        conn = tg.TigerGraphConnection(host="https://fraud-graph-kit.i.tgcloud.io", graphname="AntiFraud", username="tigergraph", password="tigergraph")
    
    conn.getToken("g8ivecpsuei7hs0hf8i0a2kcfi5oq0pl")
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
