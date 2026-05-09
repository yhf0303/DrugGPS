import os.path
import gradio as gr
import time
import random
from fontTools.merge.util import first

from model_interface import *
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from matplotlib import pyplot as plt
plt.switch_backend('Agg')


def get_tables(DataBase):
    b_dis = []
    # 这里使用SQLite作为示例，可以替换为其他数据库连接和查询
    if DataBase == "B-database":
        with open("./data/B-dataset/B_data.csv", newline='',
                  encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[2] == 'disease':
                    b_dis.append(row[1])
    elif DataBase == "F-database":
        with open("./data/F-dataset/F_data.csv", newline='',
                  encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[2] == 'disease':
                    b_dis.append(row[1])
    elif DataBase == "C-database":
        with open("./data/C-dataset/C_data.csv", newline='',
                  encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[2] == 'disease':
                    b_dis.append(row[1])
    return b_dis


def update_dropdown(database_name):
    tables = get_tables(database_name)
    return gr.update(choices=tables)


def create_network_graph(dataset_type, Mode_select,max_nodes=50):
    """
    创建训练集/测试集的网络关系图
    """
    # 这里需要根据你的实际数据结构来调整
    # 假设我们有三种节点：蛋白质、药物、疾病
    # 以及它们之间的关系

    # 创建图
    G = nx.Graph()


    # 根据数据集类型加载不同的数据
    if dataset_type == "DrugBank" or dataset_type == "DAVIS" or dataset_type == "KIBA":
        # 示例数据 - 你需要替换为实际数据加载
        protein_id_to_name = {}
        with open(f"./data/{dataset_type}/Protein_infomation.csv", 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                name = row[1]
                index = row[0]
                protein_id_to_name[index] = name
        drug_id_to_name = {}
        with open(f"./data/{dataset_type}/Drug_infomation.csv", 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                name = row[1]
                index = row[0]
                drug_id_to_name[index] = name


        # 添加节点（使用名字）
        for protein_name in protein_id_to_name.values():
            G.add_node(protein_name, type='protein', color='lightblue')
        for drug_name in drug_id_to_name.values():
            G.add_node(drug_name, type='drug', color='lightgreen')

        # 读取边关系，将编号转换为名字
        edges = []
        with open(f"./data/{dataset_type}/DrugProteinCorrelation.csv", 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            first = True
            for row in csv_reader:
                if first:
                    first = False
                    continue
                if len(row) >= 2:
                    drug_id, protein_id = row[0], row[1]
                    print(row)
                    # 将编号转换为名字
                    drug_name = drug_id_to_name[drug_id] # 如果找不到映射，使用原编号
                    protein_name = protein_id_to_name[protein_id]
                    edges.append((drug_name, protein_name))

        G.add_edges_from(edges)

    elif dataset_type == "B-dataset" or dataset_type == "C-dataset" or dataset_type == "F-dataset":
        # DAVIS数据集的数据结构
        disease_id_to_name = {}
        drug_id_to_name = {}
        protein_id_to_name = {}
        with open(f"./data/{dataset_type}/node_data.csv", 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                ID = row[2]
                index = row[0]
                if ID == "drug":
                    name = row[1]
                    drug_id_to_name[index] = name
                elif ID == "disease":
                    name = row[1]
                    disease_id_to_name[index] = name
                elif ID == "protein":
                    name = row[3]
                    protein_id_to_name[index] = name

        # 添加节点（使用名字）
        for disease_name in disease_id_to_name.values():
            G.add_node(disease_name, type='protein', color='lightcoral')
        for drug_name in drug_id_to_name.values():
            G.add_node(drug_name, type='drug', color='lightgreen')
        for protein_name in protein_id_to_name.values():
            G.add_node(protein_name, type='protein', color='lightblue')

        # 读取边关系，将编号转换为名字
        edges = []
        with open(f"./data/{dataset_type}/DrugDiseaseCorrelation.csv", 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            first = True
            for row in csv_reader:
                if first:
                    first = False
                    continue
                if len(row) >= 2:
                    drug_id, protein_id = row[0], row[1]
                    print(row)
                    # 将编号转换为名字
                    drug_name = drug_id_to_name[drug_id]
                    disease_name = disease_id_to_name[protein_id]
                    edges.append((drug_name, disease_name))
        with open(f"./data/{dataset_type}/DrugProteinCorrelation.csv", 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            first = True
            for row in csv_reader:
                if first:
                    first = False
                    continue
                if len(row) >= 2:
                    drug_id, protein_id = row[0], row[1]
                    print(row)
                    # 将编号转换为名字
                    drug_name = drug_id_to_name[drug_id]
                    protein_name = protein_id_to_name[protein_id]
                    edges.append((drug_name, protein_name))


        G.add_edges_from(edges)

    # 如果节点太多，进行采样
    if len(G.nodes()) > max_nodes:
        if Mode_select == "Display randomly":
            # 直接随机选择节点
            all_nodes = list(G.nodes())
            random.shuffle(all_nodes)
            nodes_to_keep = all_nodes[:max_nodes]
        elif Mode_select == "Display highest selectivity":
            nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            nodes_to_keep = [node for node, degree in nodes_by_degree[:max_nodes]]
        G = G.subgraph(nodes_to_keep)

    # 绘制网络图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 获取节点颜色
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    # 使用spring布局
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Protein'),
        Patch(facecolor='lightgreen', label='Drug'),
        Patch(facecolor='lightcoral', label='Disease')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(f'Network Graph - {dataset_type} Dataset\n'
                 f'Nodes: {len(G.nodes())}, Edges: {len(G.edges())}')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r"./test.png", format="png")

    return fig



with gr.Blocks(theme='', css="./web_style/style.css") as demo:
    # shivi/calm_seafoam
    gr.Image(value='./web_style/logo.png', width=200, show_download_button=False, container=False)

    with gr.Tab(label="Protein2Drug", elem_classes="body1"):
        gr.Markdown("## Protein2Drug\n\n"
            "You Need to Input Protein FASTA, Then Choose the DataBase ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                Protein = gr.Textbox(lines=5, label="Protein FASTA", elem_classes="input")
                with gr.Row():
                    DataBase = gr.Radio(["DrugBank", "DAVIS", "KIBA"], label="Choose DataBase", elem_classes="button13")
                    max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number", value=50)
                    Predict = gr.Button(value="Predict", min_width=1, variant="primary", elem_classes="button1")
                with gr.Row():
                    output_pic = gr.Plot(label="Result Image", show_label=True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=3,
                                                headers=['Drug ID', 'Score', 'SMILES'],
                                                row_count=5)
                    Predict.click(fn=TDI, inputs=[Protein, DataBase, max_num], outputs=[output_pic, table_result])

    with gr.Tab(label="Drug2Protein", elem_classes="body2"):
        gr.Markdown("## Drug2Protein\n\n"
            "You Need to Input Drug SMILES, Then Choose the DataBase ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    SMILES = gr.Textbox(lines=5, label="Drug SMILES", elem_classes="input", scale=4)
                    with gr.Column():
                        DataBase = gr.Radio(["DrugBank", "DAVIS", "KIBA"], label="Choose DataBase",
                                            elem_classes="button13")
                        max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number", value=50)
                    Predict = gr.Button(value="Predict", min_width=1, variant="primary", elem_classes="button3",
                                        scale=1)
                with gr.Row():
                    output_pic = gr.Plot(label="Result Image", show_label=True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=3,
                                                headers=['Protein ID', 'Score', 'FASTA'],
                                                row_count=5)
                    Predict.click(fn=DTI, inputs=[SMILES, DataBase, max_num], outputs=[output_pic, table_result])

    with gr.Tab(label="Disease2Drug", elem_classes="body4"):
        gr.Markdown("## Disease2Drug\n\n"
            "You Need to Choose the DataBase, Then Choose the Disease ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    DataBase = gr.Radio(["B-dataset", "C-dataset", "F-dataset"], label="Choose DataBase", scale=1)
                    Disease_list = pd.read_csv("./data/all_disease.csv", usecols=[1])
                    Disease_list = np.array(Disease_list).tolist()
                    Disease_list = [i[0] for i in Disease_list]
                    with gr.Column(scale=4):
                        Disease = gr.Dropdown(Disease_list, label="Disease Name", elem_classes="input")
                        max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number", value=50)
                    Predict = gr.Button(value="Predict", min_width=1, variant="primary", elem_classes="button4",
                                        scale=1)
                with gr.Row():
                    output_pic = gr.Plot(label="Result Image", show_label=True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=3,
                                                headers=['Drug ID', 'Score', 'SMILES'],
                                                row_count=5)
                    Predict.click(fn=Dis2Drug, inputs=[Disease, DataBase, max_num], outputs=[output_pic, table_result])

    with gr.Tab(label="Drug2Disease", elem_classes="body3"):
        gr.Markdown("## Drug2Disease\n\n"
            "You Need to Input Drug SMILES, Then Choose the DataBase ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    SMILES = gr.Textbox(lines=5, label="Drug SMILES", elem_classes="input", scale=4)
                    with gr.Column(scale=2):
                        DataBase = gr.Radio(["B-dataset", "C-dataset", "F-dataset"], label="Choose DataBase",
                                            elem_classes="button13")
                        max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number", value=50)
                    Predict = gr.Button(value="Predict", min_width=1, variant="primary", elem_classes="button2",
                                        scale=1)
                with gr.Row():
                    output_pic = gr.Plot(label="Result Image", show_label=True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=2, headers=["Disease name", "Score"],
                                                row_count=5)
                    Predict.click(fn=DDI, inputs=[SMILES, DataBase, max_num], outputs=[output_pic, table_result])

    # 新增的网络图展示界面
    with gr.Tab(label="Network Visualization", elem_classes="body5"):
        gr.Markdown("## Dataset Network Visualization\n\n"
                    "This section displays the relationships between proteins, drugs, and diseases in the training and test datasets.")
        with gr.Row():
            with gr.Column():
                dataset_select = gr.Radio(
                    ["DrugBank", "DAVIS", "KIBA","B-dataset","C-dataset","F-dataset"],
                    label="Select Dataset",
                    value="DrugBank"
                )
                with gr.Row():
                    Mode_select = gr.Radio(
                        ["Display randomly", "Display highest selectivity"],
                        label="Select Mode",
                        value="Display highest selectivity"
                    )
                    max_nodes = gr.Slider(
                        minimum=10, maximum=200, value=50, step=5,
                        label="Maximum Nodes to Display"
                    )
                    network_btn = gr.Button("Generate Network Graph", variant="primary",min_width=1,scale=1,elem_classes="button5")
                network_plot = gr.Plot(label="Network Graph")
                network_btn.click(fn=create_network_graph, inputs=[dataset_select, Mode_select, max_nodes],
                                  outputs=network_plot)



demo.launch(server_port=7860)