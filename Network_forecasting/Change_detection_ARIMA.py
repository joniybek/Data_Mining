# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


# This program is desined to detect anomalies in mobile network traffic data for Voice, SMS, GPRS and MMS&Content flows.
# It has SQLite db file as storage for scripts and traffic data
# Traffic data is queried using scripts in 'controls' table in SQLite from Oracle DB for last X days and inserted to local SQLite DB.
# After traffic is gathered for more than 8 days it can be predicted, generally it predicts using last 65 days data
# After prediction current and predicted traffic is compared and if absolute deviation is more than standard deviation of predicted model then it is counted as deviation
# Deviations are reported and then altered with prediction in internal DB
# On reporting emial is sent with deviations stating detailed information and plotting graphs with prediction and current traffic.
# In the future I will add more functions such as impact analysis, checking consistency of data in tables, checking loading issues

# All You need is to install Anaconda package, I used
import matplotlib
import sys
import sqlite3 as lite
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
#import cx_Oracle
from pandas.stats.api import ols
#%matplotlib inline
control_list=list()
df_dev=list()



# <codecell>

def connectOracle():
    conOx = cx_Oracle.connect('jakhash/')
    print conOx.version
    return conOx

#this is just for experementing with data from Oracle
def rdf(text):
    global conOx
    df=pd.read_sql(text,conOx)
    df.columns = map(str.lower, df.columns)
    return df
    

# <codecell>



# <codecell>

# this is to making exports
#ts1=pd.read_pickle('files/cont.pkl')
#ts.to_csv('example.tmp')

# <codecell>

def connectLite():
    conLite = None
    try:
        conLite = lite.connect('files\data.db')
        cur = conLite.cursor()
        cur.execute('SELECT SQLITE_VERSION()')
        data = cur.fetchone()
        print "SQLite version: %s" % data
    except lite.Error, e:
        print "Error %s:" % e.args[0]
        sys.exit(1)
    return conLite

# <codecell>

# this will read internal database given control name
def readTrend(snum,conLite):
    df=pd.read_sql("select d_timestamp,volume from trend_data where contr_id='"+snum+"' and d_timestamp>date('now','-65 days') order by date(d_timestamp) ",conLite,parse_dates='d_timestamp')
    df.columns = map(str.lower, df.columns)
    df=df.set_index('d_timestamp')
    return df
    

# <codecell>

# ARIMA model itself, in the furute will be altered with SARIMA
def fitArima(ts):
    import statsmodels.api as sm
    logged_ts = np.log(ts)
    diffed_logged_ts = (logged_ts - logged_ts.shift(7))[7:]
    p = 0
    d = 1
    q = 1
    arima = ARIMA(diffed_logged_ts, [p, d, q], exog=None, freq='D', missing='none')
    diffed_logged_results = arima.fit(trend='c', disp=False)
    predicted_diffed_logged = diffed_logged_results.predict(exog=None, dynamic=False)
    #a=pd.date_range(diffed_logged_ts.index[1], periods=90, freq='D')
    predicted_diffed_logged_ts = pd.Series(predicted_diffed_logged, index=diffed_logged_ts.index[d:])
    predicted_diffed_logged_ts = np.exp(logged_ts.shift(7) + diffed_logged_ts.shift(d) + predicted_diffed_logged_ts)
    
    concatenated = pd.concat([ts, predicted_diffed_logged_ts], axis=1, keys=['original', 'predicted'])
    #a= concatenated
    #a.plot()
    #plt.show()
    return concatenated

# <codecell>

# This will fix deviation for the purpose of accuracy in next prediction
def correct_record(df):
    global conLite
    global snum
    cur = conLite.cursor()
    cur.execute("update trend_data set volume="+str(df.predicted)+", write_flag=1 where d_timestamp='"+str(df.name)[:10]+"' and contr_id='"+snum+"'")
    conLite.commit()
    #print "update trend_data set volume="+str(df.predicted)+",write_flag=1 where d_timestamp='"+str(df.name)[:10]+"' and contr_id='"+snum+"'"
    

# <codecell>

#This is for automatically fixing deviations
def fixTrend(df):
    for i in xrange(8,len(df)):
        #x=np.std(df.iloc[:i][(df['wd']==df['wd'].iloc[i])]['original'])
        x=np.std(df.iloc[:i]['original'])
        y=df['original'].iloc[i]-df['predicted'].iloc[i]
        if np.abs(y)>3*x:
            print 'fixing data at '+str(df.iloc[i].name)
            #print str(df.iloc[i])+ '  this is y= '+str(y)+ '  this is x= '+ str(x)
            correct_record(df.iloc[i])
            break

# <codecell>

#This will insert data to internal database gived data from external source,
# Logic is so that "fixed" deviations (write_flag=1) will not be updated, new records are inserted, old ones will be updated
def updateTrend(df,snum):
    global conLite
    with conLite:
        cur = conLite.cursor()
        for i,c in df.iterrows():
            #print "insert or replace into trend_data (contr_id,d_timestamp,volume,write_flag) values ('"+snum+"','"+str(c[0])[:10]+"','"+str(c[1])+"', '0');"
            try:
                cur.execute("update or ignore trend_data set volume='"+str(c[1])+"' where contr_id='"+snum+"' and d_timestamp='"+str(c[0])[:10]+"' and write_flag=0;")
                cur.execute("insert or ignore into trend_data (contr_id,d_timestamp,volume,write_flag) values ('"+snum+"','"+str(c[0])[:10]+"','"+str(c[1])+"', '0') ;")
                conLite.commit()
            except:
                #print "insert or replace into trend_data (contr_id,d_timestamp,volume,write_flag) values ("+snum+","+c[0]+","+c[1]+", 0);"
                print sys.exc_info()[1]
        

# <codecell>

# This will make a list of current activated (active_flag) controls in internal DB, reads coresonding SQL script for each control, queries external source 
# calls fuction to update internal db with new data
def updateTrends():
    global conLite
    global conOx
    global control_list
    control_list=list()
    with conLite:
        curLite = conLite.cursor()
        curLite.execute('select contr_id, script from controls where active_flag=1')
        cs_list=[(i[0],i[1]) for i in curLite.fetchall()]
        with conOx:
            for ci,si in cs_list:
                print ci
                control_list.append(ci)
                df=df=pd.read_sql(si,conOx)
                try:
                    updateTrend(df,ci)
                except:
                    print 'error in updating trends'
    return control_list
 

# <codecell>

# This will check deviations in predicted and current model, mismatches is added to massive
def checkTrend(cnum,df):
    deviations=list()
    df['wd']=df.index.weekday
    df.columns = df.columns.get_level_values(0)
                
    for i in xrange(len(df)-15,len(df)):
        #std=np.std(df.iloc[:i][(df['wd']==df['wd'].iloc[i])]['original'])
        #mean=np.mean(df.iloc[:i][(df['wd']==df['wd'].iloc[i])]['original'])
        std=np.std(df.iloc[:i]['predicted'])
        #mean=np.mean(df.iloc[:i]['predicted'])
        diff=df['original'].iloc[i]-df['predicted'].iloc[i]
        #print str(df.iloc[i].name)+' at original v= ' +str(df['original'].iloc[i]) +' predicted v= '+str(df['predicted'].iloc[i])+' standard dev= '+str(std)+' diff= '+str(diff)
        if np.abs(diff)>std:
            deviations.append((df.iloc[i].name,cnum,diff,std,df['original'].iloc[i],df['predicted'].iloc[i]))
            #print diff,std
            #print 'there is deviation at '+str(df.iloc[i].name)
            #print str(df.iloc[i])+ '  this is y= '+str(y)+ '  this is x= '+ str(x)+ '  this is = '+ str(x)
    return deviations
   

# <codecell>

# This is for sending email through Unix server, Multipart email is prepared in Windows machine locally and uploaded to Unix and sent via Mutt client.
# Multipart email includes: Deviation with detailed information and Plotted graph saved as jpg file
# This part is under further development
def sendMailViaTetra(txt,attachments,recievers):
    import paramiko
    from paramiko import SSHClient
    from scp import SCPClient
    host='xxx'
    user='xxx'
    psec='xxx'
    port=22
    folder='scripts/jakhashr/email'
    file_='temp.msg'
    sshclient = paramiko.SSHClient()
    sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sshclient.connect(hostname=host, username=user, password=psec, port=port)
    scp = SCPClient(sshclient.get_transport())
    scp.put(file_, folder)
    #stdin, stdout, stderr = sshclient.exec_command('ls -l')
    #data = stdout.read() + stderr.read()
    #print data
    #client.close
    
    
    
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    from email.MIMEImage import MIMEImage
    strFrom = 'from@example.com'
    strTo = 'jakhongir.ashrapov@tele2.com'
    
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'test message'
    msgRoot['From'] = strFrom
    msgRoot['To'] = strTo
    msgRoot.preamble = 'This is a multi-part message in MIME format.'
    alternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    
    msgText = MIMEText('This is the alternative plain text message.')
    msgAlternative.attach(msgText)
    # We reference the image in the IMG SRC attribute by the ID we give it below
    msgText = MIMEText('<b>Some <i>HTML</i> text</b> and an image.<br><img src="cid:image1"><br>Nifty!', 'html')
    msgAlternative.attach(msgText)
    
    # Image file should be created already as temp.jpg
    fp = open('temp.jpg', 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()
    
    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)
    f=open('temp.m','w')
    f.write(msgRoot.as_string())
    f.close()

    

# <codecell>

# this will create HTML email with detailed information
def reportDeviation(ci,df,deviations):
    from string import Template
    def readControlMeta():
        global conLite
        
        
        
    plotfile=ci+'.jpg'
    today=datetime.now().strftime("%Y-%m-%d %H:%M")
    df.plot(ci)
    plt.savefig(plotfile)
    
    html="""<b> $date1 </b> and an image.<br><img src="cid:image1"><br>Nifty! """
    
    

    
    
    

# <codecell>

conLite=connectLite() 
conOx=connectOracle() 
control_list=updateTrends()  #this is for updating all trends

# <codecell>

# this will fit active controls with queried data, creates massive with all models and deviations and plots into graph, saves and calls
# method to send email
import threading
for ci in control_list:

    snum=ci
    ts=readTrend(ci, conLite)
    if len(ts)>15:
        df=fitArima(ts)
        #deviations=checkTrend(ci,df)
        df.plot(title=ci)
        #plt.savefig('temp.jpg') # currently for testing only
        t1=threading.Thread(target=plt.show())
        t1.start()
        for i in deviations:
            print i
        df_dev.append((ci,df,deviations))
    

    
    

# <codecell>

## this is for my comp
conLite=connectLite() 
control_list=list()
with conLite:
    curLite = conLite.cursor()
    curLite.execute('select contr_id, script from controls where active_flag=1')
    cs_list=[(i[0],i[1]) for i in curLite.fetchall()]
    for ci,si in cs_list:
        print ci
        control_list.append(ci)

# <codecell>

for ci in control_list:
    snum=ci
    ts=readTrend(ci, conLite)
    if len(ts)>15:
        df=fitArima(ts)
        #deviations=checkTrend(ci,df)
        df.plot(title=ci)
        #plt.savefig('temp.jpg') # currently for testing only
        t1=threading.Thread(target=plt.show())
        t1.start()

#manually correct ubnormal trends
#correct_record(df.loc['2015-02-23']) # example of manual fixing deviations given data

#save output of each graph as file
a=plt.savefig(ci+'.jpg')

