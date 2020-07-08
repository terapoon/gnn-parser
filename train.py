import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from modules import Model
from lib import load_data
from lib import get_info
from lib import layer_wise_loss
from lib import MST
from lib import assert_edges


train_path = 'UD_English-EWT/en_ewt-ud-train.conllu'
test_path = 'UD_English-EWT/en_ewt-ud-test.conllu'
n_epoch = 1


def _main():
    os.makedirs('results', exist_ok=True)

    print('loading data...')
    train_data, test_data, word_size = load_data(train_path, test_path)
    tag_size = 18

    print('word id: {}, tag id:{}'.format(word_size, tag_size))

    train_data, valid_data = train_test_split(train_data, test_size=0.25)
    train_len = len(train_data)
    valid_len = len(valid_data)
    train_inputs, train_tags, train_edges, train_n_vertices, _ = get_info(train_data)
    valid_inputs, valid_tags, valid_edges, valid_n_vertices, _ = get_info(valid_data)

    device = torch.device("cuda")
    model = Model(word_size, tag_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.9))

    train_loss_log = []
    train_acc_log = []
    valid_loss_log = []
    valid_acc_log = []

    print('training starts')
    for epoch in range(n_epoch):
        train_total_loss = 0.0
        train_acc_num = 0.0
        valid_total_loss = 0.0
        valid_acc_num = 0.0
        print('epoch', epoch + 1)

        print('training...')
        for inputs, tags, edges, n_vertices in zip(tqdm(train_inputs), train_tags, train_edges, train_n_vertices):
            inputs = inputs.to(device)
            tags = tags.to(device)
            # edge_labels = get_edge_labels(data)
            optimizer.zero_grad()
            head, dependent, alpha1, alpha2, alpha3 = model(inputs, tags)
            loss = layer_wise_loss(alpha1, alpha2, alpha3, n_vertices, edges)
            loss.backward()
            optimizer.step()
            mst_edges = MST(alpha3)
            if assert_edges(edges, mst_edges):
                train_acc_num += 1.0
            train_total_loss += loss
        train_acc = train_acc_num / train_len
        train_loss_mean = train_total_loss / train_len
        print("train loss mean={}, train acc={}".format(train_acc, train_loss_mean))
        train_loss_log.append(train_loss_mean)
        train_acc_log.append(train_acc)

        print('validating...')
        for inputs, tags, edges, n_vertices in zip(tqdm(valid_inputs), valid_tags, valid_edges, valid_n_vertices):
            # edge_labels = get_edge_labels(data)
            inputs.to(device)
            tags.to(device)
            optimizer.zero_grad()
            head, dependent, alpha1, alpha2, alpha3 = model(inputs, tags)
            loss = layer_wise_loss(alpha1, alpha2, alpha3, n_vertices, edges)
            mst_edges = MST(alpha3)
            if assert_edges(edges, mst_edges):
                valid_acc_num += 1.0
            valid_total_loss += loss
        valid_acc = valid_acc_num / valid_len
        valid_loss_mean = valid_total_loss / valid_len
        print("valid loss mean={}, valid acc={}".format(valid_acc, valid_loss_mean))
        valid_loss_log.append(valid_loss_mean)
        valid_acc_log.append(valid_acc)

    plt.figure(figsize=(6, 6))
    plt.plot(range(n_epoch), train_loss_log)
    plt.plot(range(n_epoch), valid_loss_log, c='#00ff00')
    plt.xlim(0, n_epoch)
    plt.ylim(0, 2.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'valid loss'])
    plt.title('loss')
    plt.savefig("results/loss_image.png")
    plt.clf()

    plt.plot(range(n_epoch), train_acc_log)
    plt.plot(range(n_epoch), valid_acc_log, c='#00ff00')
    plt.xlim(0, n_epoch)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'valid acc'])
    plt.title('accuracy')
    plt.savefig("results/accuracy_image.png")


if __name__ == '__main__':
    _main()
