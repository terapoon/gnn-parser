import collections
import torch
import networkx as nx

from settings import UPOSTAG_IDX_DICT
from settings import DEPREL_IDX_DICT


def parse_data(data_path):
    flag = False
    data = []
    word_list = []

    with open(data_path, 'r') as f:
        while (True):
            newline = f.readline()
            if not newline:
                break

            parsed = newline.split()
            if parsed == []:
                if flag:
                    n_vertices = len(new_sentence.keys())
                    data.append((n_vertices, new_sentence))
                    flag = False
                    continue

            elif parsed[0] == '#':
                if parsed[1] == 'text':
                    flag = True
                    new_sentence = {}
                    continue

            else:
                if parsed[6] == '_':
                    flag = False

                else:
                    info = {
                        'LEMMA': parsed[2], 'UPOSTAG': parsed[3],
                        'HEAD': int(parsed[6]), 'DEPREL': parsed[7].split(":")[0]
                    }
                    new_sentence[int(parsed[0])] = info
                    word_list.append(parsed[2])
    return word_list, data


def make_word_id_dict(word_count, border=3):
    idx = 1
    word_id = {}
    for word in word_count.keys():
        if word_count[word] <= border:
            word_id[word] = 0
        else:
            word_id[word] = idx
            idx += 1
    word_id['ROOT'] = idx + 1
    return word_id, idx + 2


def process_data(word_list, data):
    processed_data = []
    word_count = collections.Counter(word_list)
    word_id_dict, word_size = make_word_id_dict(word_count)
    for (n_vertices, sentence) in data:
        processed_n_vertices = n_vertices + 1
        processed_sentence = {}
        processed_sentence[0] = {
            'LEMMA': word_id_dict['ROOT'], 'UPOSTAG': UPOSTAG_IDX_DICT['ROOT'],
            'HEAD': 0, 'DEPREL': DEPREL_IDX_DICT['nop']
        }
        processed_edges = []
        for i in range(1, processed_n_vertices):
            processed_sentence[i] = {
                'LEMMA': word_id_dict[sentence[i]['LEMMA']],
                'UPOSTAG': UPOSTAG_IDX_DICT[sentence[i]['UPOSTAG']],
                'HEAD': sentence[i]['HEAD'],
                'DEPREL': DEPREL_IDX_DICT[sentence[i]['DEPREL']]
            }
            processed_edges.append((sentence[i]['HEAD'], i))
        processed_data.append((processed_n_vertices, processed_edges, processed_sentence))
    return processed_data, word_size


def load_data(train_path, test_path):
    word_list_train, unprocessed_train_data = parse_data(train_path)
    word_list_test, unprocessed_test_data = parse_data(test_path)
    word_list = word_list_train + word_list_test
    processed_train_data, word_size= process_data(word_list, unprocessed_train_data)
    processed_test_data, _ = process_data(word_list, unprocessed_test_data)
    return processed_train_data, processed_test_data, word_size


def get_info(dataset):
    inputs_list = []
    tags_list = []
    edges_list = []
    n_vertices_list = []
    labels_list = []
    for data in dataset:
        inputs, tags, edges, n_vertices, labels = get_info_from_data(data)
        inputs_list.append(inputs)
        tags_list.append(tags)
        edges_list.append(edges)
        n_vertices_list.append(n_vertices)
        labels_list.append(labels)
    return inputs_list, tags_list, edges_list, n_vertices_list, labels_list


def get_info_from_data(data):
    inputs = []
    tags = []
    n_vertices = data[0]
    labels = []
    edges = data[1]
    for i in range(data[0]):
        inputs.append(data[2][i]['LEMMA'])
        tags.append(data[2][i]['UPOSTAG'])
    for e in edges:
        labels.append(data[2][e[1]]['DEPREL'])
    return torch.tensor([inputs]), torch.tensor([tags]), edges, n_vertices, torch.tensor([labels])


def get_edge_labels(data):
    labels = []
    edges = data[1]
    for e in edges:
        labels.append(data[2][e[1]]['DEPREL'])
    return torch.tensor(labels)


def layer_wise_loss(alpha1, alpha2, alpha3, n_vertices, edges):
    l1 = 0
    l2 = 0
    l3 = 0
    for e in edges:
        l1 += torch.log(alpha1[e[0]][e[1]]+1e-12)
        l2 += torch.log(alpha2[e[0]][e[1]]+1e-12)
        l3 += torch.log(alpha3[e[0]][e[1]]+1e-12)
    return -(l1 + l2 + l3) / n_vertices


def MST(alpha):
    G = nx.DiGraph()
    n_vertices = len(alpha)
    for i in range(n_vertices):
        G.add_node(i)
    for i in range(n_vertices):
        for j in range(n_vertices):
            G.add_edge(i, j, weight=alpha[i][j])
    edges = list(nx.maximum_spanning_arborescence(G))
    return edges


def assert_edges(edges1, edges2):
    edge_set1 = set(edges1)
    edge_set2 = set(edges2)
    return edge_set1 == edge_set2
