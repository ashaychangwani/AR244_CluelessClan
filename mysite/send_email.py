import email
import smtplib
import ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from datetime import datetime
import random

import glob
import os


def email_alert():
    
    print('sending email')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    factory_id = random.randint(1,5)
    production_line = random.randint(1,10)
    batch_id = random.randint(1,500)

    subject = "Defect found in Fabric Line"
    body = "Dear admin,\nA defect has been detected at "+dt_string+" at factory number "+str(factory_id)+"in the production line "+str(production_line)+" in batch number "+str(batch_id)+". \n\nRegards,\nAuto-Alert System"
    sender_email = "pratik.devnani98@gmail.com"
    receiver_email = "ashaychangwani@gmail.com"
    password = "Monster@0255"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    # message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    files1 = glob.glob('C:\\Users\\Ashay\\front\\mysite\\static\\images\\*.png')
    
    for f in files1:
        filename = f
        img_data = open(filename, 'rb').read()
        image = MIMEImage(img_data, name=os.path.basename(filename[-15:0]))
        message.attach(image)
        text = message.as_string()
        # # Open PDF file in binary mode
        # with open(filename, "rb") as attachment:
        #     # Add file as application/octet-stream
        #     # Email client can usually download this automatically as attachment
        #     part = MIMEBase("application", "octet-stream")
        #     part.set_payload(attachment.read())

        # # Encode file in ASCII characters to send by email
        # encoders.encode_base64(part)

        # # Add header as key/value pair to attachment part
        # part.add_header(
        #     "Content-Disposition",
        #     f"attachment; filename= {filename}",
        # )

        # # Add attachment to message and convert message to string
        # message.attach(part)
        

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    print('middle')
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)
        print('sent')
    return
    
def error_email(img_path):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    factory_id = random.randint(1, 5)
    production_line = random.randint(1, 10)
    batch_id = random.randint(1, 500)

    subject = "Error in production line"
    body = "Dear admin,\nAn unidentified input has been detected at "+dt_string+" at factory number " + \
        str(factory_id)+" in the production line "+str(production_line) + \
        " in batch number "+str(batch_id)+". \n\nRegards,\nAuto-Alert System"
    sender_email = "pratik.devnani98@gmail.com"
    receiver_email = "mkljngd@gmail.com"
    password = "*********"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    # message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))
    img_data = open(img_path, 'rb').read()
    image = MIMEImage(img_data, name=os.path.basename(img_path))
    message.attach(image)
    text = message.as_string()
      
    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    return
