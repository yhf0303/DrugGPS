import os.path

import gradio as gr
import time
from model_interface import *
import csv


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

# def get_result(Drug, DataBase, max_num,progress=gr.Progress(track_tqdm=True)):
#     path = "D:\BaiduNetdiskDownload\AI素材\ch219-人工智能\ch219-jpg图片"
#     imgs_List = [(os.path.join(path , name),name) for name in sorted(os.listdir(path))]
#     max_num = pd.read_csv("H:\Code_reproduction\\2012.04231-Modof-main\\result.csv")
#     for i in range(10):
#         time.sleep(1)  # 模拟长时间运行的任务
#         progress(i/10, total=10)  # 更新进度条
#     return imgs_List, max_num



with gr.Blocks(theme='',css="./web_style/style.css") as demo:
    # shivi/calm_seafoam
    gr.Image(value='./web_style/logo.png',width=200,show_download_button=False,container=False)
    # gr.Markdown(value ="<img src='logo.png'/>",elem_id="imge")
    with gr.Tab(label="Protein2Drug",elem_classes="body1"):
        gr.Markdown("You Need to Input Protein FASTA, Then Choose the DataBase ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                Protein = gr.Textbox(lines=5, label="Protein FASTA",elem_classes="input")
                with gr.Row():
                    DataBase = gr.Radio(["DrugBank", "DAVIS", "KIBA"], label="Choose DataBase",elem_classes="button13")
                    max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number",value=50)
                    Predict = gr.Button(value="Predict", min_width=1,variant="primary",elem_classes="button1")
                with gr.Row():
                    # value = max_num.change(fn=process_slider_value, inputs=max_num, outputs=output))
                    output_pic = gr.Plot(label="Result Image",show_label = True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=3, headers=['Drug ID', 'Score', 'SMILES'],
                                                row_count=5)
                    Predict.click(fn=TDI, inputs=[Protein, DataBase, max_num], outputs=[output_pic, table_result])
    with gr.Tab(label="Drug2Protein",elem_classes="body2"):
        gr.Markdown("You Need to Input Drug SMILES, Then Choose the DataBase ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    SMILES = gr.Textbox(lines=5, label="Drug SMILES",elem_classes="input",scale = 4)
                    with gr.Column():
                        DataBase = gr.Radio(["DrugBank", "DAVIS", "KIBA"], label="Choose DataBase",elem_classes="button13")
                        max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number",value=50)
                    Predict = gr.Button(value="Predict", min_width=1,variant="primary",elem_classes="button3",scale = 1 )
                with gr.Row():
                    # value = max_num.change(fn=process_slider_value, inputs=max_num, outputs=output))
                    output_pic = gr.Plot(label="Result Image",show_label = True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=3, headers=['Protein ID', 'Score','FASTA'],
                                                row_count=5)
                    Predict.click(fn=DTI, inputs=[SMILES, DataBase, max_num], outputs=[output_pic, table_result])
    with gr.Tab(label="Disease2Drug",elem_classes="body4"):
        gr.Markdown("You Need to Choose the DataBase, Then Choose the Disease ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    DataBase = gr.Radio(["B-dataset", "C-dataset", "F-dataset"], label="Choose DataBase",scale = 1)
                    Disease_list = pd.read_csv("./data/all_disease.csv", usecols=[1])
                    Disease_list = np.array(Disease_list).tolist()
                    Disease_list = [i[0] for i in Disease_list]
                    with gr.Column(scale=4):
                        Disease = gr.Dropdown(Disease_list, label="Disease Name", elem_classes="input")
                        max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number", value=50)
                    # DataBase.change(update_dropdown, inputs=DataBase, outputs=Disease)
                    Predict = gr.Button(value="Predict", min_width=1, variant="primary", elem_classes="button4",scale = 1)
                with gr.Row():
                    # value = max_num.change(fn=process_slider_value, inputs=max_num, outputs=output))
                    output_pic = gr.Plot(label="Result Image",show_label = True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=3, headers=['Drug ID', 'Score', 'SMILES'],
                                                row_count=5)
                    Predict.click(fn=Dis2Drug, inputs=[Disease, DataBase, max_num], outputs=[output_pic, table_result])
    with gr.Tab(label="Drug2Disease",elem_classes="body3"):
        gr.Markdown("You Need to Input Drug SMILES, Then Choose the DataBase ,Last Press Predict. So That You Can Get the Relation Result")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    SMILES = gr.Textbox(lines=5, label="Drug SMILES", elem_classes="input", scale=4)
                    with gr.Column(scale=2):
                        DataBase = gr.Radio(["B-dataset", "C-dataset", "F-dataset"], label="Choose DataBase",
                                            elem_classes="button13")
                        max_num = gr.Slider(minimum=0, maximum=100, step=1, label="Show maximum number", value=50)
                    Predict = gr.Button(value="Predict", min_width=1, variant="primary", elem_classes="button2",scale=1)
                with gr.Row():
                    # value = max_num.change(fn=process_slider_value, inputs=max_num, outputs=output))
                    output_pic = gr.Plot(label="Result Image",show_label = True)
                    table_result = gr.DataFrame(label="Predict Result", col_count=2, headers=["Disease name", "Score"],
                                                row_count=5)
                    Predict.click(fn=DDI, inputs=[SMILES, DataBase, max_num], outputs=[output_pic, table_result])



demo.launch(server_port = 7860)
