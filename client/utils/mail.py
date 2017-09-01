#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.header import Header

def send_mail(inputs):
    sender = 'from@runoob.com'
    receivers = ['cbbupt@163.com','cbbupt@163.com']

    message = MIMEText(inputs, 'plain', 'utf-8')
    message['From'] = Header("belief", 'utf-8')
    message['To'] = Header("cb", 'utf-8')

    subject = 'Real Time Loss'
    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtpObj = smtplib.SMTP('0.0.0.0')
        smtpObj.sendmail(sender, receivers, message.as_string())
        print "send cbbupt@163.com mail successfully"
    except smtplib.SMTPException:
        print "Error: can't send mail"

if __name__ == '__main__':
    inputs='kjhsd\nasidjiosdj'
    send_mail(inputs)