import torch
import os
from pyvis.network import Network

def visualize_rgnn_graph_pyvis(dataset_path, output_file='graph.html', max_nodes=100):
    """
    RGNNのグラフデータをPyVisで可視化する関数
    
    Args:
        dataset_path: processed_dataset.ptへのパス
        output_file: 保存するHTMLファイル名
        max_nodes: 可視化する最大ノード数
    """
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: File not found at {dataset_path}")
        return

    # データセットの読み込み
    data = torch.load(dataset_path)
    vocab = data['vocab']
    # IDから単語への逆マッピング
    idx_to_word = {v: k for k, v in vocab.items()}

    # 全データ（Train/Val/Test）のエッジを統合
    splits = ['train', 'val', 'test']
    all_edge_index = []
    all_edge_type = []

    for split in splits:
        if split in data:
            all_edge_index.append(data[split]['pos_edge_index'])
            all_edge_type.append(data[split]['pos_edge_type'])

    if not all_edge_index:
        print("No edges found in the dataset.")
        return

    edge_index = torch.cat(all_edge_index, dim=1)
    edge_types = torch.cat(all_edge_type)

    print(f"Total edges: {edge_index.shape[1]}")
    print(f"Visualizing subgraph with approx {max_nodes} nodes...")

    # PyVisネットワークの初期化
    # directed=Trueで有向グラフにする
    # 背景を白に変更 (font_colorはadd_nodeで個別に指定するため削除)
    net = Network(height="750px", width="100%", bgcolor="white", directed=True)
    
    # 物理シミュレーションの設定
    # 中心に集まるように重力を調整
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=95, spring_strength=0.001, damping=0.09, overlap=0)

    # 可視化対象とするノードIDのセット（単純に0からmax_nodesまで）
    target_nodes = set(range(min(max_nodes, len(vocab))))

    # ノードの追加
    for node_id in target_nodes:
        label = idx_to_word.get(node_id, str(node_id))
        # titleはマウスオーバー時に表示される
        # sizeはデフォルト、文字サイズを大きく(40)、色は黒
        net.add_node(node_id, label=label, title=f"ID: {node_id}", font={'size': 40, 'color': 'black'})

    # エッジの追加
    edge_count = 0
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        etype = edge_types[i].item()

        # 両方のノードが対象範囲内にある場合のみ追加
        if src in target_nodes and dst in target_nodes:
            # Type 1 (ID 0): Composition -> Solid line
            # Type 2 (ID 1): Term Link -> Dotted line
            
            if etype == 0:
                # Solid line
                net.add_edge(src, dst, color="#4ad0ff", width=2, title="Composition Link")
            elif etype == 1:
                # Dotted line
                # PyVisでは dashes=True で破線/点線になります
                net.add_edge(src, dst, color="#ff5e5e", width=2, dashes=True, title="Term Link")
            
            edge_count += 1

    print(f"Added {len(target_nodes)} nodes and {edge_count} edges.")

    # 設定用ボタンを表示（任意）
    # net.show_buttons(filter_=['physics'])

    # HTMLとして保存
    # ノートブック環境でない場合は open_browser=False 推奨だが、
    # ローカル実行を想定してそのまま保存します。
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    try:
        net.save_graph(output_path)
        print(f"Graph saved to {output_path}")
    except Exception as e:
        print(f"Error saving graph: {e}")

if __name__ == '__main__':
    # データパスの解決
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # デフォルトパス: ../data/debug_dataset.pt
    default_data_path = os.path.join(current_dir, '..', 'data', 'debug_dataset.pt')
    default_data_path = os.path.normpath(default_data_path)
    
    visualize_rgnn_graph_pyvis(default_data_path, output_file='debug_graph.html', max_nodes=50)

