import numpy as np
import torch
from torch.utils.data import Dataset
from pyproj import Transformer as projTransformer
import pdb

np.random.seed(2333)

class SpatialDataset(Dataset):
    def __init__(self, tokenizer , max_token_len ,  distance_norm_factor, sep_between_neighbors = False ):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len 
        self.distance_norm_factor = distance_norm_factor
        self.sep_between_neighbors = sep_between_neighbors
        self.ptransformer = projTransformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=True) # https://epsg.io/4087, equidistant cylindrical projection
        

    def parse_spatial_context(self, pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill,  pivot_dist_fill = 0):

        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        mask_token_id  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        max_token_len = self.max_token_len


        # process pivot
        # 将地名转换为token ID序列
        # 示例：pivot_name="北京" -> tokens=['北', '京'] -> IDs=[1234, 5678]
        pivot_name_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pivot_name))
        pivot_token_len = len(pivot_name_tokens)
            
        pivot_lng = pivot_pos[0]
        pivot_lat = pivot_pos[1]

        # 将经纬度转换为投影坐标
        pivot_lng, pivot_lat =  self.ptransformer.transform(pivot_lng, pivot_lat)  

        # prepare entity mask
        entity_mask_arr = []
        #为所有实体（中心地名+邻居地名）生成随机数决定是否掩码
        rand_entity = np.random.uniform(size = len(neighbor_name_list) + 1) # random number for masking entities including neighbors and pivot
        # True for mask, False for unmask
        
        # check if pivot entity needs to be masked out, 15% prob. to be masked out
        if rand_entity[0] < 0.15:
            entity_mask_arr.extend([True] * pivot_token_len)
        else:
            entity_mask_arr.extend([False] * pivot_token_len)

        # process neighbors
        neighbor_token_list = []
        neighbor_lng_list = []
        neighbor_lat_list = []

        # add separator between pivot and neighbor tokens
        # checking pivot_dist_fill is a trick to avoid adding separator token after the class name (for class name encoding of margin-ranking loss)
        if self.sep_between_neighbors and pivot_dist_fill==0: 
            neighbor_lng_list.append(spatial_dist_fill)
            neighbor_lat_list.append(spatial_dist_fill)
            neighbor_token_list.append(sep_token_id)

        # 邻居地名处理
        for neighbor_name, neighbor_geometry, rnd in zip(neighbor_name_list, neighbor_geometry_list, rand_entity[1:]):

            # 只考虑以字母开头的邻居
            if not neighbor_name[0].isalpha():
                # only consider neighbors starting with letters
                continue 
            
            # 将邻居地名转换为token ID序列
            neighbor_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(neighbor_name))
            neighbor_token_len = len(neighbor_token)

            # compute the relative distance from neighbor to pivot,
            # normalize the relative distance by distance_norm_factor
            # apply the calculated distance for all the subtokens of the neighbor
            # neighbor_lng_list.extend([(neighbor_geometry[0]- pivot_lng)/self.distance_norm_factor] * neighbor_token_len)
            # neighbor_lat_list.extend([(neighbor_geometry[1]- pivot_lat)/self.distance_norm_factor] * neighbor_token_len)

            #坐标投影变换
            if 'coordinates' in neighbor_geometry: # to handle different json dict structures
                neighbor_lng , neighbor_lat = self.ptransformer.transform(
                    neighbor_geometry['coordinates'][0],
                    neighbor_geometry['coordinates'][1]
                )  
                
            else:
                neighbor_lng , neighbor_lat = self.ptransformer.transform(neighbor_geometry[0], neighbor_geometry[1])  

            # 计算邻居到中心地名的相对距离，并归一化
            neighbor_lng_list.extend([(neighbor_lng - pivot_lng)/self.distance_norm_factor] * neighbor_token_len)
            neighbor_lat_list.extend([(neighbor_lat - pivot_lat)/self.distance_norm_factor] * neighbor_token_len)
            neighbor_token_list.extend(neighbor_token)


            if self.sep_between_neighbors:
                neighbor_lng_list.append(spatial_dist_fill)
                neighbor_lat_list.append(spatial_dist_fill)
                neighbor_token_list.append(sep_token_id)
                
                entity_mask_arr.extend([False])

            # 邻居地名也有15%的概率被掩码
            if rnd < 0.15:
                #True: mask out, False: Keey original token
                entity_mask_arr.extend([True] * neighbor_token_len)
            else:
                entity_mask_arr.extend([False] * neighbor_token_len)

        # 中心地名tokens + 邻居地名tokens
        pseudo_sentence = pivot_name_tokens + neighbor_token_list 
        
        # 构建距离列表：中心地名位置用0填充，邻居用相对距离
        dist_lng_list = [pivot_dist_fill] * pivot_token_len + neighbor_lng_list 
        dist_lat_list = [pivot_dist_fill] * pivot_token_len + neighbor_lat_list 
        

        ## 计算句子长度（不含CLS和SEP）
        #including cls and sep
        sent_len = len(pseudo_sentence)
        max_token_len_middle = max_token_len -2 # 2 for CLS and SEP token

        # padding and truncation
        if sent_len > max_token_len_middle : 
            pseudo_sentence = [cls_token_id] + pseudo_sentence[:max_token_len_middle] + [sep_token_id] 
            dist_lat_list = [spatial_dist_fill] + dist_lat_list[:max_token_len_middle]+ [spatial_dist_fill]
            dist_lng_list = [spatial_dist_fill] + dist_lng_list[:max_token_len_middle]+ [spatial_dist_fill]
            attention_mask = [0] + [1] * max_token_len_middle + [0] # make sure SEP and CLS are not attented to
        else:
            pad_len = max_token_len_middle - sent_len
            assert pad_len >= 0 

            pseudo_sentence = [cls_token_id] + pseudo_sentence + [sep_token_id] + [pad_token_id] * pad_len 
            dist_lat_list = [spatial_dist_fill] + dist_lat_list + [spatial_dist_fill] + [spatial_dist_fill] * pad_len
            dist_lng_list = [spatial_dist_fill] + dist_lng_list + [spatial_dist_fill] + [spatial_dist_fill] * pad_len
            attention_mask = [0] + [1] * sent_len + [0] * pad_len + [0]



        norm_lng_list = np.array(dist_lng_list) 
        norm_lat_list = np.array(dist_lat_list) 


        # mask entity in the pseudo sentence 
        #实体掩码，直接掩码完整地名实体
        entity_mask_indices = np.where(entity_mask_arr)[0] # true: mask out
        masked_entity_input = [mask_token_id if i in entity_mask_indices else pseudo_sentence[i] for i in range(0, max_token_len)]

        # mask token in the pseudo sentence
        rand_token = np.random.uniform(size = len(pseudo_sentence))
        # do not mask out cls and sep token. True: masked tokens False: Keey original token
        #token掩码，掩码单个token
        token_mask_arr = (rand_token <0.15) & (np.array(pseudo_sentence) != cls_token_id) & (np.array(pseudo_sentence) != sep_token_id) & (np.array(pseudo_sentence) != pad_token_id)
        token_mask_indices = np.where(token_mask_arr)[0]
        
        masked_token_input = [mask_token_id if i in token_mask_indices else pseudo_sentence[i] for i in range(0, max_token_len)]
        

        ## 50%概率选择实体掩码，50%概率选择token掩码
        if np.random.rand() > 0.5:
            masked_input = torch.tensor(masked_entity_input)
        else:
            masked_input = torch.tensor(masked_token_input)
        
        train_data = {}
        # train_data['pivot_name'] = pivot_name
        ## 中心地名在句子中的位置
        train_data['pivot_token_idx'] = torch.tensor([1,pivot_token_len+1])
        #掩码后的输入
        train_data['masked_input'] = masked_input
        #句子位置ID
        train_data['sent_position_ids'] = torch.tensor(np.arange(0, len(pseudo_sentence)))
        #注意力掩码 
        train_data['attention_mask'] = torch.tensor(attention_mask)
        #经度距离归一化列表
        train_data['norm_lng_list'] = torch.tensor(norm_lng_list).to(torch.float32)
        #纬度距离归一化列表
        train_data['norm_lat_list'] = torch.tensor(norm_lat_list).to(torch.float32)
        #原始句子，用于计算损失
        train_data['pseudo_sentence'] = torch.tensor(pseudo_sentence)
        #token类型ID，0表示自然语言，1表示空间数据
        train_data['token_type_ids'] = torch.zeros(len(pseudo_sentence)).int()

        return train_data



    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError