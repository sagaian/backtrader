
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[9]:

'''
check for NaN
'''
def isNaN(num):
    return num != num

'''
Get min, max, mean from two n-day intervals before and after trigger events.
Input: signal column name - String
       target column name - String
       data - DataFrame
       n - integer

Output: Two DataFrame containing statistics, [min, mean, max]
        lists of actual data of target columns within n days, the current data is included
'''
def get_stats_around_triggers(signal_col,target_col,data,n):
    stats_b = []
    stats_a = []
    val = []
    min_b = []
    min_a = []
    mean_b = []
    mean_a = []
    max_b = []
    max_a = []
    ind = []
    for i in range(data.shape[0]):
        if (isNaN(data[signal_col][i])):
            continue
        if (data[signal_col][i] != 0):
            stats_temp_a = []
            stats_temp_b = []
            if (i < n):
                min_a.append(np.min(data[target_col][(i+1):(i+n+1)]))
                mean_a.append(np.average(data[target_col][(i+1):(i+n+1)]))
                max_a.append(np.max(data[target_col][(i+1):(i+n+1)]))
                stats_a.append(data[target_col][(i+1):(i+n+1)].tolist())
                #stats_a.append(stats_temp_a)
                
                min_b.append(np.min(data[target_col][:i]))
                mean_b.append(np.average(data[target_col][:i]))
                max_b.append(np.max(data[target_col][:i]))
                
                ind.append(data.index[i])
                val.append(data[target_col][i])
                stats_a.append(data[target_col][i:(i+n+1)].tolist())
                stats_b.append(data[target_col][:(i+1)].tolist())
                #stats_b.append(stats_temp_b)
            else:
                min_a.append(np.min(data[target_col][(i+1):(i+n+1)]))
                mean_a.append(np.average(data[target_col][(i+1):(i+n+1)]))
                max_a.append(np.max(data[target_col][(i+1):(i+n+1)]))
                #stats_a.append(stats_temp_a)
                
                min_b.append(np.min(data[target_col][(i-n):i]))
                mean_b.append(np.average(data[target_col][(i-n):i]))
                max_b.append(np.max(data[target_col][(i-n):i]))
                
                ind.append(data.index[i])
                val.append(data[target_col][i])
                stats_a.append(data[target_col][i:(i+n+1)].tolist())
                stats_b.append(data[target_col][(i-n):(i+1)].tolist())
                #stats_b.append(stats_temp_b)


    df_b = pd.DataFrame({'min':min_b,'mean':mean_b,'max':max_b,target_col:val},
                        index=ind,columns=["min","mean","max",target_col])
    df_a = pd.DataFrame({'min':min_a,'mean':mean_a,'max':max_a,target_col:val},
                        index=ind,columns=["min","mean","max",target_col])
    
    return df_b, df_a, stats_b, stats_a                     


# In[7]:

'''
Functions in this block will mark signals. If other signals are need, please add them on your own. 
However, use backtrader or Excel to generate signals are much faster. 
'''

'''
Count number of trigger events
Input: Data - DataFrame
       Event Name - String
Output: Number of Events - Integer
'''
def get_trigger_nums(df_tia,col):
    ct = 0
    for i in range(df_tia.shape[0]):
        if (df_tia[col][i] != 0):
            ct += 1
    return ct

'''
This function will mark overbought and oversold for the input and will create two columns corresponding to OB and OS.
Input: Data - DataFrame
Output: DataFrame with additional columns representing overbought and oversold.
'''
def mark_obos(df_ti):
    ob = [0]*df_ti.shape[0]
    os = [0]*df_ti.shape[0]
    for i in range(df_ti.shape[0]):
        if (df_ti["rsi"][i] > 70):
            ob[i] = 90
        elif (df_ti["rsi"][i] < 30):
            os[i] = 90
    df_ti["OB"] = ob
    df_ti["OS"] = os
    return df_ti

'''
This function will mark cross up and down for the input and will create two columns corresponding to them.
Input: Data - DataFrame
Output: DataFrame with additional columns.
'''
def mark_crosses(df_ti,signal_1,signal_2,day1,day2):
    up = [0]*df_ti.shape[0]
    dn = [0]*df_ti.shape[0]
    for i in range(df_ti.shape[0]):
        diff = df_ti[signal_1][i] - df_ti[signal_2][i]
        diff_prev = df_ti[signal_1][i-1] - df_ti[signal_2][i-1]
        if (diff <= 0 and diff_prev > 0):
            dn[i] = 90
        elif (diff >= 0 and diff_prev < 0):
            up[i] = 90
    df_ti["up_"+signal_1+"_"+day1+"_"+day2] = up
    df_ti["down_"+signal_1+"_"+day1+"_"+day2] = dn
    return df_ti

'''
This function will mark all break outs of Bollinger Bands
'''
def mark_BBands_Breaks(df_ti,bot,top,close):
    top_up = [0]*df_ti.shape[0]
    bot_dn = [0]*df_ti.shape[0]
    
    for i in range(df_ti.shape[0]):
        diff_top = df_ti[top][i] - df_ti[close][i]
        diff_top_prev = df_ti[top][i-1] - df_ti[close][i-1]
        diff_bot = df_ti[bot][i] - df_ti[close][i]
        diff_bot_prev = df_ti[bot][i-1] - df_ti[close][i-1]
        if (diff_top >= 0 and diff_top_prev < 0):
            top_up[i] = 90
        elif (diff_bot <= 0 and diff_bot_prev > 0):
            bot_dn[i] = 90
    df_ti["bbands_break_up"] = top_up
    df_ti["bbands_break_down"] = bot_dn
    return df_ti


# In[8]:

'''
Convert input DataFrame into Yahoo Finance style with all data as the target column
input: DataFrame
       int, target column index
output: DataFrame
'''
def to_Yahoo_Finance(df,col_ind):
    col = ["Open","High","Low","Close","Adj Close","Volume"]
    df_temp = pd.DataFrame(index=df.index)
    for i in col:
        df_temp[i] = df.iloc[:,col_ind]
    return df_temp

'''
Return the ratios as a data frame
input: DataFrame, data before and after the signal
output: DataFrame, ratios
'''
def get_summary_table(dfb,dfa):
    _min = []
    _max = []
    _mean = []
    for i in range(dfb.shape[0]):
        _min.append(dfa["min"][i]/dfb["min"][i])
        _max.append(dfa["max"][i]/dfb["max"][i])
        _mean.append(dfa["mean"][i]/dfb["mean"][i])
    return pd.DataFrame({"min":_min,"mean":_mean,"max":_max},index=dfb.index)

'''
Concat DataFrames
'''
def concat_df(df_tia,signal_col,target_col,day_interval):
    b_1,a_1,vb_1,va_1 = get_stats_around_triggers(signal_col,target_col,df_tia,day_interval[0])
    sum_table = get_summary_table(b_1,a_1)
    df = pd.DataFrame(index=b_1.index)
    for i in range(sum_table.shape[1]):
        df[sum_table.columns[i]+"_"+str(day_interval[0])] = sum_table[sum_table.columns[i]]
        
    for i in day_interval[1:]:
        obb,oba,vixb,vixa = get_stats_around_triggers(signal_col,target_col,df_tia,i)
        temp_sum = get_summary_table(obb,oba)
        for j in range(temp_sum.shape[1]):
            df[temp_sum.columns[j]+"_"+str(i)] = temp_sum[temp_sum.columns[j]]
    return df

'''
Determine if this signal is strong
input: Dict
output: boolean
'''
def is_strong_signal(n_dict,cur_n_dict):
    for k,v in n_dict.items():
        if (cur_n_dict[k] < n_dict[k]):
            return False
    return True

'''
target_stats_list: min, mean, or max
target_day_intervals: list of day intervals interested in
n: a list of n. need n "useful" stats (min,mean,max) to meet requirements to be considered as "strong" signal
threshold: need changes greater than the threshold to be considerred as 1 "useful" stats
'''
def get_strong_signals(df_avg,target_stats_list,target_day_intervals,n_dict,threshold):
    strong_list = []
    col_name_list = []
    for target_stats in target_stats_list:
        for cur_day_interval in target_day_intervals:
            col_name_list.append(target_stats+"_"+str(cur_day_interval))
    for ind in df_avg.index:
        cur_n_dict = dict(zip(n_dict.keys(),np.zeros(len(n_dict))))
        for cur_col in col_name_list:
            stats_name,di = cur_col.split("_")
            if (df_avg[cur_col][ind] >= 1+threshold or df_avg[cur_col][ind] <= 1-threshold):
                cur_n_dict[stats_name] += 1
        if (is_strong_signal(n_dict,cur_n_dict)):
            strong_list.append(ind)
    df = pd.DataFrame(index=strong_list,columns=df_avg.columns)
    for ind in df.index:
        for col in df.columns:
            df[col][ind] = df_avg[col][ind]
    return df

def make_line_text(name,content):
    tx = name+": ["
    for i in content:
        tx += i+","
    tx = tx[:-1]
    tx += "]\n"
    return tx

def make_text(target_day_intervals,n_dict,min_chg):
    target_stats, n_val = zip(*n_dict.items())
    n_val_str = [str(i) for i in n_val]
    day_interval_str = [str(i) for i in target_day_intervals]
    tx = make_line_text("target stats",target_stats)
    tx += make_line_text("target day interval",day_interval_str)
    tx += make_line_text("min strong stats",n_val_str)
    tx += "min change = "+str(min_chg)
    return tx
        
'''
Export Summary Table as Excel File
'''
def export_summary_table(day_interval,df_tia,signal_col_list,target_col,dir_path,filename,target_stats_list,
                        target_day_intervals,n,threshold):
    writer = pd.ExcelWriter(dir_path+filename+".xlsx",engine='xlsxwriter')
    ob_temp_1 = concat_df(df_tia,signal_col_list[0],target_col,day_interval)
    df_avg = pd.DataFrame(index=signal_col_list, columns=ob_temp_1.columns)
    temp_col = []
    flag = 0
    for signal_col in signal_col_list:
        ob_temp = concat_df(df_tia,signal_col,target_col,day_interval)
        for i in range(ob_temp.shape[1]):
            df_avg[ob_temp.columns[i]][signal_col] = ob_temp[ob_temp.columns[i]].mean()
        ob_temp.to_excel(writer,signal_col)
    df_strong_signal = get_strong_signals(df_avg,target_stats_list,target_day_intervals,n,threshold)
    df_avg.to_excel(writer,"Average Summary")
    df_strong_signal.to_excel(writer,"Strong Signals")
    workbook = writer.book
    worksheet = workbook.add_worksheet("Thresholds")
    threshold_text = make_text(target_day_intervals,n,threshold)
    row = col = 1
    worksheet.insert_textbox(row, col, threshold_text)
    writer.save()
    
'''
read data and set date as index and sorted by the index
input: string, file type, csv or excel
       string, file path
       int, date column index starting from 0
output: DataFrame
'''
def read_data(file_type,path,date_ind):
    if (file_type == "csv"):
        df = pd.read_csv(path,index_col=date_ind)
    elif (file_type == "excel"):
        df = pd.read_excel(path,index_col=date_ind)
    else:
        print("Wrong File Type!")
        return
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

