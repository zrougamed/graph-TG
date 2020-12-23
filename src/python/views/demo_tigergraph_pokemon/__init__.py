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

from gremlin_python import statics
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import WithOptions, T

############################################
#
#   DASHBOARD SETTINGS
#
############################################
#  Controls how entrypoint.py picks it up


app_id = 'tigergraph_Covid' 
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


# Define the name of the view
def info():
    return {
        'id': app_id,
        'name': 'TigerGraph: Pokemon Filter',
        'tags': ['demo', 'tigergraph_demo_Pokemon']
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
    conn = tg.TigerGraphConnection(host="http://172.21.0.1", graphname="MyGraph", username="tigergraph", password="tigergraph",useCert=False)
    q = conn.runInstalledQuery("mostDirectInfections")
    Most_infectious_IDS = q[0]['Answer']
    MI_List = [d['v_id'] for d in Most_infectious_IDS if 'v_id' in d]
    num_edges_init = urlParams.get_field('num_matches', 50)
    MI_List.reverse()
    patient_id = st.sidebar.selectbox(
        'Patient ID ',
        MI_List
        )

    # city = st.sidebar.text_input(
    #     'Find ',
    #     "")

    num_edges = st.sidebar.slider(
        'Infection depth', min_value=1, max_value=100, value=num_edges_init, step=1)
    urlParams.set_field('num_edges', num_edges)
    urlParams.set_field('patient_id', patient_id)

    return {'num_edges': num_edges, 'patient_id': patient_id}


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

def type_to_color(t):
    mapper = {'Patient': 0xFF000000}
    if t in mapper:
        return mapper[t]
    else:
        return 0xFFFFFF00

# Given filter settings, generate/cache/return dataframes & viz
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def run_filters(num_edges, patient_id):
    global metrics
        
    conn = tg.TigerGraphConnection(host="http://172.21.0.1", graphname="MyGraph", username="tigergraph", password="tigergraph",useCert=False)
    
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

    q = conn.runInstalledQuery("infectionSubgraph",{"p":patient_id,"depthSize":num_edges})
    edges = q[0]['@@edgeSet']

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


def main_area(url, nodes, edges, patient_id):

    logger.debug('rendering main area, with url: %s', url)
    GraphistrySt().render_url(url)
    try:
        conn = tg.TigerGraphConnection(host="http://172.21.0.1", graphname="MyGraph", username="tigergraph", password="tigergraph",useCert=False)
        q = conn.runInstalledQuery("ageDistribution")
        ageMap = q[0]['@@ageMap']
        # cleaning the N/A Ages
        del ageMap["2020"]
    except Exception as e:
        print(e)
    
    # Get the count by patient_id of visits shown
    bar_chart_data = pd.DataFrame.from_dict(ageMap,  orient = 'index')
    st.bar_chart(bar_chart_data)
    #Show  a datatable with the values transposed
    #st.dataframe(bar_chart_data.set_index(group_label).T)

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
                      sidebar_filters['patient_id'])
        else:  # render a message
            st.write("No data matching the specfiied criteria is found")

    except Exception as exn:
        st.write('Error loading dashboard')
        st.write(exn)
