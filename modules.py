import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.33):
        super(ArcMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, lstm_embedding_dim=100, lstm_hidden_dim=400, lstm_num_layers=3, lstm_dropout=0.33,
                 mlp_hidden_dim=1024, mlp_output_dim=500, mlp_dropout=0.33):
        super(Encoder, self).__init__()
        self.lstm_embedding_dim = lstm_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_output_dim = mlp_output_dim
        self.mlp_dropout = mlp_dropout
        self.bilstm = nn.LSTM(
            self.lstm_embedding_dim, self.lstm_hidden_dim, num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout, bidirectional=True
        )
        self.mlp_head = ArcMLP(
            self.lstm_hidden_dim, self.mlp_hidden_dim, self.mlp_output_dim, dropout=self.mlp_dropout
        )
        self.mlp_dependent = ArcMLP(
            self.lstm_hidden_dim, self.mlp_hidden_dim, self.mlp_output_dim, dropout=self.mlp_dropout
        )

    def forward(self, x):
        x, (_, _) = self.bilstm(x)
        splited_x = torch.chunk(x, 2, dim=2)
        head = splited_x[0][0]
        dependent = splited_x[1][0]
        head = self.mlp_head(head)
        dependent = self.mlp_dependent(dependent)
        return head, dependent


class SynchronizedSoftGNN(nn.Module):
    def __init__(self, input_dim):
        super(SynchronizedSoftGNN, self).__init__()
        self.input_dim = input_dim
        self.agg_weight_head = nn.Parameter(torch.rand(self.input_dim, self.input_dim))
        self.agg_weight_dependent = nn.Parameter(torch.rand(self.input_dim, self.input_dim))
        self.comb_weight_head = nn.Parameter(torch.rand(self.input_dim, self.input_dim))
        self.comb_weight_dependent = nn.Parameter(torch.rand(self.input_dim, self.input_dim))
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def _aggregate(self, head, dependent, alpha):
        head_agg = torch.zeros(head.shape).cuda()
        dependency_agg = torch.zeros(dependent.shape).cuda()
        n_vertices = len(alpha)
        for i in range(n_vertices):
            for j in range(n_vertices):
                if j != i:
                    head_agg[i] += alpha[j][i]*head[j]
                    head_agg[i] += alpha[i][j]*dependent[j]
                    dependency_agg[i] += alpha[i][j]*head[j]
                    dependency_agg[i] += alpha[j][i]*dependent[j]
        return head_agg, dependency_agg

    def forward(self, head, dependent, alpha):
        head_agg, dependent_agg = self._aggregate(head, dependent, alpha)
        head_agg = torch.mm(head_agg, self.agg_weight_head)
        dependent_agg = torch.mm(dependent_agg, self.agg_weight_dependent)
        head_comb = torch.mm(head, self.comb_weight_head)
        dependent_comb = torch.mm(dependent, self.comb_weight_dependent)
        head = self.LeakyReLU(head_agg + head_comb)
        dependent = self.LeakyReLU(dependent_agg + dependent_comb)
        return head, dependent


class TransProb(nn.Module):
    def __init__(self, input_dim):
        super(TransProb, self).__init__()
        self.input_dim = input_dim
        self.A = nn.Parameter(torch.rand(self.input_dim, self.input_dim))
        self.b1 = nn.Parameter(torch.rand(self.input_dim))
        self.b2 = nn.Parameter(torch.rand(self.input_dim))
        self.Softmax = nn.Softmax(1)

    def forward(self, head, dependent):
        dual = torch.mm(torch.mm(head, self.A), torch.t(dependent))
        linear1 = torch.unsqueeze(torch.mv(head, self.b1), 1)
        linear2 = torch.mv(dependent, self.b2)
        x = dual + linear1 + linear2
        alpha = self.Softmax(x)
        return alpha


class GNNLayer(nn.Module):
    def __init__(self, input_dim):
        super(GNNLayer, self).__init__()
        self.input_dim = input_dim
        self.transprob1 = TransProb(self.input_dim)
        self.transprob2 = TransProb(self.input_dim)
        #self.transprob3 = TransProb(self.input_dim)
        self.gnn1 = SynchronizedSoftGNN(self.input_dim)
        self.gnn2 = SynchronizedSoftGNN(self.input_dim)
        #self.gnn3 = SynchronizedSoftGNN(self.input_dim)

    def forward(self, head, dependent):
        alpha1 = self.transprob1(head, dependent)
        head, dependent = self.gnn1(head, dependent, alpha1)
        alpha2 = self.transprob2(head, dependent)
        head, dependent = self.gnn2(head, dependent, alpha2)
        #alpha3 = self.transprob3(head, dependent)
        #head, dependent = self.gnn3(head, dependent, alpha3)
        #return head, dependent, alpha1, alpha2, alpha3
        return head, dependent, alpha1, alpha2


class Model(nn.Module):
    def __init__(
            self, word_size, tag_size, word_embedding_dim=100, tag_embedding_dim=100, lstm_hidden_dim=400,
            lstm_num_layers=3, lstm_dropout=0.33, mlp_hidden_dim=1024, mlp_output_dim=500, mlp_dropout=0.33
    ):
        super(Model, self).__init__()
        self.word_size = word_size
        self.tag_size = tag_size
        self.word_embedding_dim = word_embedding_dim
        self.tag_embedding_dim = tag_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_output_dim = mlp_output_dim
        self.mlp_dropout = mlp_dropout
        self.word_embedding = nn.Embedding(self.word_size, self.word_embedding_dim)
        self.tag_embedding = nn.Embedding(self.tag_size, self.tag_embedding_dim)
        self.encoder = Encoder(
            self.word_embedding_dim+self.tag_embedding_dim, self.lstm_hidden_dim, self.lstm_num_layers,
            self.lstm_dropout, self.mlp_hidden_dim, self.mlp_output_dim, self.mlp_dropout
        )
        self.gnn_layer = GNNLayer(self.mlp_output_dim)
        # RelMLP layer
        # self.relmlp = ...

    def forward(self, word, tag):
        word_embeds = self.word_embedding(word)
        tag_embeds = self.tag_embedding(tag)
        embeds = torch.cat((word_embeds, tag_embeds), 2)
        head, dependent = self.encoder(embeds)
        #head, dependent, alpha1, alpha2, alpha3 = self.gnn_layer(head, dependent)
        head, dependent, alpha1, alpha2 = self.gnn_layer(head, dependent)
        # RelMLP layer
        # tag = self.rel_nlp(...)
        #return head, dependent, alpha1, alpha2, alpha3
        return head, dependent, alpha1, alpha2
