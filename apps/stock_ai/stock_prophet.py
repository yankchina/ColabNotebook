import csv
import requests
from datetime import date
import pandas as  pd
from prophet import Prophet


## TODO: Typing Check
def execute(stock_id,start_date,end_date,output_path):
    ## Download Stock Data From Website
    stock_name = u""
    url_string = u"http://quotes.money.163.com/service/chddata.html?code={0}&start={1}&end={2}".format(stock_id,start_date,end_date)
    df = pd.DataFrame(columns=['ds','y'])
    with requests.Session() as s:
        download =  s.get(url_string)
        decoded_content = download.content.decode(u"GB2312")
        cr = csv.reader(decoded_content.splitlines(),delimiter=u',')
        data_list = list(cr)
        if len(data_list) == 1:
            ## TODO: Exception
            return 
        for i,row in enumerate(data_list):
            if i == 0:
                continue
            if i > 0:
                df = df.append({'ds':row[0],'y':row[3]},ignore_index=True)
            if i == 1:
                stock_name = row[2]
    df['ds'] = pd.to_datetime(df['ds'],format='%Y-%m-%d')
    
    ## Predict 
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365) 
    forcast = model.predict(future)
    
    ## Save Result
    predict_result_fig1_file_name = u"{0}/fig1_{1}_{2}.png".format(output_path,stock_id,end_date)
    predict_result_fig2_file_name = u"{0}/fig2_{1}_{2}.png".format(output_path,stock_id,end_date)
    predict_result_csv_file_name = u"{0}/predict_{1}_{2}.csv".format(output_path,stock_id,end_date)
    fig1 = model.plot(forcast)
    fig1.savefig(predict_result_fig1_file_name)
    fig2 = model.plot_components(forcast)
    fig2.savefig(predict_result_fig2_file_name)
    forcast.to_csv(predict_result_csv_file_name, encoding=u"utf-8")
    
    ## End
    print(u"end{0}-{1}".format(stock_id,stock_name))
    pass