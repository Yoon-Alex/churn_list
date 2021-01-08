#!/usr/bin/env python
# coding: utf-8

#===========================================================
# File Name : B_DA_RETL_CHURN
# Description :                                             
# Date :                                    
# Writer : Yoon Jun Beom
# Packages :                                                
# Note :                           
#===========================================================

# library import 
import os, sys
import jaydebeapi as jdb
import pandas as pd
import numpy as np
import seaborn as sns
import datetime, time
import time
import joblib

from sklearn.ensemble import RandomForestClassifier

def churn_base(conn): 
    sql = """
    SELECT  *
      FROM  NEW_BIGDATA.B_DA_RETL_CHURN_ANL_MD_DS
    """
    
    df = pd.read_sql(sql, conn)
    return df[['reg_no','amt_day_cnt', 'avg_amt', 'avg_cnt', 'avg_mm_amt', 'avg_mm_cnt',
               'bf_month_diff_amt', 'bf_month_diff_cnt', 'high_lw_term',
               'join_term', 'last_xmit_date', 'm3_sd_amt', 'm3_sd_cnt', 'm_00_app_amt',
               'm_00_app_cnt', 'm_01_app_amt', 'm_01_app_cnt', 'm_02_app_amt',
               'm_02_app_cnt', 'm_03_app_amt', 'm_03_app_cnt', 'm_04_app_amt',
               'm_04_app_cnt', 'm_05_app_amt', 'm_05_app_cnt', 'max_amt_month',
               'min_amt_month', 'min_mm_amt', 'min_mm_cnt', 'min_xmit_date',
               'month_cnt', 'mx_mm_amt', 'mx_mm_cnt', 'mx_xmit_date', 
               'sd_amt', 'sd_mm_amt', 'sd_mm_cnt', 'sum_amt', 'sum_cnt',
               'wk2_amt_diff', 'wk2_cnt_diff']]


def reg_table(conn): 
    sql = """
    SELECT  A.REG_NO 
            , B.MAP_CD
            , B.CD_LV2_NM
      FROM  NEW_BIGDATA.DEBIT_RETL A 
      LEFT 
      JOIN  (
            SELECT  MAP_CD 
                    , CD_LV2_NM
              FROM  NEW_BIGDATA.MAP_CODE_INFO
             WHERE  MAP_GUBUN == 'V'
            ) B 
        ON  A.DBR_NEW_SVR_CODE = B.MAP_CD    
    """
    df = pd.read_sql(sql, conn)
    
    li =     [["부페", "중식", "퓨전/기타음식점", "한식-일반음식점", "양식", "일식/수산물", "분식/휴게음식점", "패스트푸드/제과점"]	, "음식점" ,
    ["실내스포츠시설운영업", "실외스포츠시설운영업", "스포츠시설운영업"]	, "스포츠시설운영업",
    ["예체능계학원", "외국어학원", "일반교과학원", "기타학원", "학습지/코칭교육"]	, "학원",
    ["자동차및부품판매점", "자동차정비/세차/주차장"]	, "자동차관련판매점",
    ["유흥주점-무도/가무", "일반유흥주점"]	, "주점",
    ["독서실/도서관"]	, "기타시설운영업",
    ["세탁/가사/전문서비스", "주택수리서비스", "용품수리서비스"]	, "대행/용역/인력알선",
    ["약국/한약방", "유사의료업", "의료관련서비스업"]	, "병원",
    ["예식/의례/관혼상제"]	, "수의업"]    
    
    a = []  
    b = []

    for i in li:
        if type(i) == list :
            a.append(i)
        else :
            b.append(i)

    for i, j in zip(a, b):
        df['cd_lv2_nm'].replace(i, j, inplace= True)        
    
    return df


def recent_sales(conn, args_mm): 
    # 가맹점 최근 실적
    sql = \ 
    f"""
    SELECT  SUBSTRING(XMIT_DATE, 1, 6) XMIT_MONTH
            , REG_NO        
            , SUM(APP_CNT) APP_CNT
            , SUM(APP_AMT) APP_AMT
      FROM  (
            SELECT  SOS_XMIT_DATE XMIT_DATE
                    , SOS_REG_NO REG_NO
                    , SOS_COMM_CNT APP_CNT
                    , SOS_COMM_AMT APP_AMT
              FROM  NEW_BIGDATA.ST_ONLINE_SUM_DAY A
             WHERE  SUBSTRING(A.SOS_XMIT_DATE, 1, 6) = {args_mm}
                    AND SUBSTR(A.SOS_AGENCY_CODE, 1, 1) != 'X'                   
             UNION   
               ALL
            SELECT  A.SCS_XMIT_DATE XMIT_DATE
                    , A.SCS_REG_NO REG_NO
                    , A.SCS_COMM_CNT APP_CNT
                    , A.SCS_COMM_AMT APP_AMT
              FROM  NEW_BIGDATA.ST_CASH_SUM_DAY A
             WHERE  SUBSTRING(A.SCS_XMIT_DATE, 1, 6) = {args_mm}
                    AND A.SCS_COMM_AMT not in ( 1004,  999111999 )
                    AND SUBSTR(A.SCS_AGENCY_CODE, 1, 1) != 'X'
            ) A
     GROUP 
        BY  SUBSTRING(XMIT_DATE, 1, 6) 
            , REG_NO                       
    """
    
    df = pd.read_sql(sql, conn)


def reg_info(conn): 
    sql = """
    SELECT  A.REG_NO 
            , B.MAP_CD
            , LEFT(A.H_CODE, 5)
            , B.CD_LV2_NM
            , C.SIDO_NM 
            , C.GUNGU_NM 
      FROM  NEW_BIGDATA.DEBIT_RETL A 
      LEFT 
      JOIN  (
            SELECT  MAP_CD 
                    , CD_LV2_NM
              FROM  NEW_BIGDATA.MAP_CODE_INFO
             WHERE  MAP_GUBUN == 'V'
            ) B 
        ON  A.DBR_NEW_SVR_CODE = B.MAP_CD    
      LEFT 
      JOIN  NEW_BIGDATA.B_HCODE_GUNGU C
    """
    df = pd.read_sql(sql, conn)
    
    li =     [["부페", "중식", "퓨전/기타음식점", "한식-일반음식점", "양식", "일식/수산물", "분식/휴게음식점", "패스트푸드/제과점"]	, "음식점" ,
    ["실내스포츠시설운영업", "실외스포츠시설운영업", "스포츠시설운영업"]	, "스포츠시설운영업",
    ["예체능계학원", "외국어학원", "일반교과학원", "기타학원", "학습지/코칭교육"]	, "학원",
    ["자동차및부품판매점", "자동차정비/세차/주차장"]	, "자동차관련판매점",
    ["유흥주점-무도/가무", "일반유흥주점"]	, "주점",
    ["독서실/도서관"]	, "기타시설운영업",
    ["세탁/가사/전문서비스", "주택수리서비스", "용품수리서비스"]	, "대행/용역/인력알선",
    ["약국/한약방", "유사의료업", "의료관련서비스업"]	, "병원",
    ["예식/의례/관혼상제"]	, "수의업"]    
    
    a = []  
    b = []

    for i in li:
        if type(i) == list :
            a.append(i)
        else :
            b.append(i)

    for i, j in zip(a, b):
        df['cd_lv2_nm'].replace(i, j, inplace= True)        
    
    return df

def preprocess(month_tot, debit_retl):
    month_tot['reg_no'] = month_tot.reg_no.astype("str")
    
    for i in range(6):
        month_tot[f'm{str(i+1)}_cnt_rto'] = month_tot[f'm_{str(i).zfill(2)}_app_cnt'] / month_tot['sum_cnt'] 
        month_tot[f'm{str(i+1)}_amt_rto'] = month_tot[f'm_{str(i).zfill(2)}_app_amt'] / month_tot['sum_amt'] 

    for i in range(5):        
        month_tot[f'm{str(i).zfill(2)}_cnt_updn_rto'] = (month_tot[f'm_{str(i).zfill(2)}_app_cnt'] / month_tot[f'm_{str(i+1).zfill(2)}_app_cnt']) -1 
        month_tot[f'm{str(i).zfill(2)}_amt_updn_rto'] = (month_tot[f'm_{str(i).zfill(2)}_app_amt'] / month_tot[f'm_{str(i+1).zfill(2)}_app_amt']) -1 

    month_tot.replace([np.inf, -np.inf], np.nan, inplace = True)
    
    for col in month_tot.columns : # [month_tot.columns.str.contains(r"(_rto)")]
        if month_tot[col].dtype == 'object': 
            pass 
        month_tot.fillna({col:0}, inplace = True)
        
    month_tot['avg_cnt_updn_rto'] = month_tot[month_tot.columns[month_tot.columns.str.contains(r"(_cnt_updn_rto)")]].mean(1)
    month_tot['avg_amt_updn_rto'] = month_tot[month_tot.columns[month_tot.columns.str.contains(r"(_amt_updn_rto)")]].mean(1)

    base_tot = month_tot.merge(debit_retl[['reg_no','cd_lv2_nm']], 'left', on = 'reg_no')
    base_tot = base_tot[base_tot.cd_lv2_nm.notna()]

    # 운영했던 일자 수
    base_tot['norm_wrk_day'] = pd.to_datetime(base_tot.mx_xmit_date, format = "%Y%m%d") - pd.to_datetime(base_tot.min_xmit_date, format = "%Y%m%d") + np.timedelta64(1, 'D')
    base_tot['norm_wrk_day'] = base_tot.norm_wrk_day / np.timedelta64(1, 'D')
    
    # 매출일 수 / 운영일 수
    base_tot['amt_day_rto'] = base_tot.amt_day_cnt / base_tot.norm_wrk_day    
    reg_churn_model = base_tot.drop(["mx_xmit_date", "last_xmit_date","min_xmit_date", "norm_wrk_day", "amt_day_cnt",
                      "min_mm_amt", "mx_mm_amt", "max_amt_month", "min_amt_month", "mx_mm_cnt", "min_mm_cnt"], axis = 1)        
    
    reg_churn_model['churn_yn'] = reg_churn_model.churn_yn.factorize()[0]
    reg_churn_model['cd_lv2_nm'] = reg_churn_model.cd_lv2_nm.factorize()[0]
    reg_churn_model['cd_lv2_nm_f'] = reg_churn.cd_lv2_nm.factorize()[0]
    
    return reg_churn_model

if __name__ == "__main__":
    conn = jdb.connect('','',['[id]', '[password]'],'[jar file]')
    
    # get parameter
    args_mm = sys.argv[1]

    # check the parameters
    if len(args_mm) != 6 : 
        print("arg's length is not 6 char <yyyymm>")
        raise Error

    month_tot = churn_base(conn)
    debit_retl = reg_table(conn)
    
    # 전처리
    reg_churn = preprocess(month_tot, debit_retl)
    
    # 모델 Load
    rf = joblib.load('churn_model.pkl')
    X = reg_churn.drop(['reg_no','churn_yn'], axis = 1).values
        
    reg_churn['result'] = rf.predict_proba(X)
    reg_churn['grade'] = np.digitize(reg_churn['result'], [0, 0.15, 0.35, 0.5, 1.0], right = False)
    reg_churn['grade']
    
    recent_sale = recent_sales(conn, args_mm)
    reg_info = reg_info(conn)
    df = recent_sale.merge(reg_info, 'left', on = 'reg_no')
    
    # save table
    df.to_sql('B_DA_RETL_CHURN', con = conn, if_exists= 'replace', chunksize = 1000, index=False)    

