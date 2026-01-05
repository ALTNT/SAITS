from modeling.layers import *
from modeling.utils import masked_mae_cal

class SAITS_for_CACM(nn.Module):
    def __init__(
        self,
        n_groups,#5
        n_group_inner_layers,#1
        d_time,#48
        d_feature,#37
        d_model,#256
        d_inner,#512
        n_head,#8
        d_k,#32
        d_v,#32
        dropout,#0.0
        **kwargs#{'device': 'cuda', 'MIT': True, 'input_with_mask': True, 'diagonal_attention_mask': True, 'param_sharing_strategy': 'inner_group'}
    ):
        super().__init__()
        self.n_groups = n_groups#5
        self.n_group_inner_layers = n_group_inner_layers#1
        self.input_with_mask = kwargs["input_with_mask"]#True
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature#37 * 2 =74 类似于输入的波段数
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]#"inner_group"
        self.MIT = kwargs["MIT"]#True
        self.device = kwargs["device"]#'cuda'

        if kwargs["param_sharing_strategy"] == "between_group":#False 在“组”之间共享参数。只创建一套“组内层”（长度为 n_group_inner_layers ），前向时把这套层重复执行 n_groups 次,前向时外层循环按组重复执行同一堆栈,此时参数最少（同一套参数用于所有组）
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
            # self.layer_stack_for_second_block = nn.ModuleList(
            #     [
            #         EncoderLayer(
            #             d_time,
            #             actual_d_feature,
            #             d_model,
            #             d_inner,
            #             n_head,
            #             d_k,
            #             d_v,
            #             dropout,
            #             0,
            #             **kwargs
            #         )
            #         for _ in range(n_group_inner_layers)
            #     ]
            # )
        else: #True # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,#75
                        actual_d_feature,#74
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)#n_groups = 5
                ]
            )
            # self.layer_stack_for_second_block = nn.ModuleList(
            #     [
            #         EncoderLayer(
            #             d_time,
            #             actual_d_feature,#74
            #             d_model,
            #             d_inner,
            #             n_head,
            #             d_k,
            #             d_v,
            #             dropout,
            #             0,
            #             **kwargs
            #         )
            #         for _ in range(n_groups)#n_groups = 5
            #     ]
            # )

        self.dropout = nn.Dropout(p=dropout)
        # self.position_enc = PositionalEncoding(d_model, n_position=d_time)#d_model = 256, d_time = 48
        #
        self.position_enc = PositionalEncoding(d_model, 366+2, 10000)#d_model = 256, d_time = 48
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)#actual_d_feature = 74, d_model = 256
        self.reduce_dim_z = nn.Linear(d_model, d_feature)#d_model = 256, d_feature = 37
        # for the 2nd block
        # self.embedding_2 = nn.Linear(actual_d_feature, d_model)#actual_d_feature = 74, d_model = 256
        # self.reduce_dim_beta = nn.Linear(d_model, d_feature)#d_model = 256, d_feature = 37
        # self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)#d_feature = 37
        # for the 3rd block
        # self.weight_combine = nn.Linear(d_feature + d_time, d_feature)#d_feature + d_time =37+48= 75

    def impute(self, inputs):
        X, masks, doy = inputs["X"], inputs["missing_mask"], inputs["doy"]#torch.Size([128, 48, 37]) torch.Size([128, 48, 37]) torch.Size([128, 48])
        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X#torch.Size([128, 48, 74])
        input_X_for_first = self.embedding_1(input_X_for_first)#torch.Size([128, 48, 256]) nn.Linear
        enc_output = self.dropout(
            self.position_enc(input_X_for_first, doy)
        )  # namely term e in math algo torch.Size([128, 48, 256])
        if self.param_sharing_strategy == "between_group":#False
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:#inner_group
            for encoder_layer in self.layer_stack_for_first_block:#n_groups = 5
                for _ in range(self.n_group_inner_layers):#self.n_group_inner_layers=1
                    enc_output, _ = encoder_layer(enc_output)#torch.Size([128, 48, 256])

        # X_tilde_1 = self.reduce_dim_z(enc_output)#torch.Size([128, 48, 37]) 只是个线性层
        learned_presentation = self.reduce_dim_z(enc_output)
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation
        # X_prime = masks * X + (1 - masks) * X_tilde_1#torch.Size([128, 48, 37])

        # the second DMSA block
        # input_X_for_second = (
        #     torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        # )#torch.Size([128, 48, 74])
        # input_X_for_second = self.embedding_2(input_X_for_second)#torch.Size([128, 48, 256])
        # enc_output = self.position_enc(
        #     input_X_for_second
        # )  # namely term alpha in math algo
        # if self.param_sharing_strategy == "between_group":#False
        #     for _ in range(self.n_groups):
        #         for encoder_layer in self.layer_stack_for_second_block:
        #             enc_output, attn_weights = encoder_layer(enc_output)
        # else:
        #     for encoder_layer in self.layer_stack_for_second_block:##n_groups = 5
        #         for _ in range(self.n_group_inner_layers):#self.n_group_inner_layers=1
        #             enc_output, attn_weights = encoder_layer(enc_output)#torch.Size([128, 48, 256]) torch.Size([128, 8, 48, 48])

        # X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))#torch.Size([128, 48, 37])

        # the attention-weighted combination block
        # attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo torch.Size([128, 8, 48, 48])
        # if len(attn_weights.shape) == 4:#True
        #     # if having more than 1 head, then average attention weights from all heads
        #     attn_weights = torch.transpose(attn_weights, 1, 3)#torch.Size([128, 48, 48, 8])
        #     attn_weights = attn_weights.mean(dim=3)#torch.Size([128, 48, 48])
        #     attn_weights = torch.transpose(attn_weights, 1, 2)#torch.Size([128, 48, 48])

        # combining_weights = F.sigmoid(
        #     self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        # )  # namely term eta torch.Size([128, 48, 37])
        # # combine X_tilde_1 and X_tilde_2
        # # 动态地 融合两个中间插值结果（$X̃_1$ 和 $X̃_2$），以生成更准确的最终插值结果 $X̃_3$
        # # $X̃_1$（来自第一个 DMSA 块） ：侧重于利用 原始观测数据 和时间位置信息进行初步插值。
        # # $X̃_2$（来自第二个 DMSA 块） ：基于 $X̃_1$ 的结果再次进行自注意力计算，捕捉更深层的依赖关系，即“基于插值的插值”。
        # # 两者可能在不同的时间点或特征上各有优劣。简单的平均（0.5 + 0.5）无法发挥这种互补性。
        # X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1#torch.Size([128, 48, 37])
        # # replace non-missing part with original data
        # X_c = masks * X + (1 - masks) * X_tilde_3#torch.Size([128, 48, 37])#缺失值用X_tilde_3填充，非缺失值用X填充
        # return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):#stage='Train'
        X, masks = inputs["X"], inputs["missing_mask"]#torch.Size([128, 48, 37]) torch.Size([128, 48, 37])
        # reconstruction_loss = 0
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)

        # reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        # reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        # final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        # reconstruction_loss += final_reconstruction_MAE
        # reconstruction_loss /= 3

        if (self.MIT or stage == "val") and stage != "test":#True
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                learned_presentation, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }