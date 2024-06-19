# emailAlert.py

import smtplib
from email.message import EmailMessage
import os

def email_alert(subject, body, to, photo_path):
    message = EmailMessage()
    message.set_content(body)
    message['subject'] = subject
    message['to'] = to

    user = "9221WacoWay@gmail.com"
    message['from'] = user
    password = "fqzg mxab dvku kaof"

    # Attach the photo
    if os.path.exists(photo_path):
        with open(photo_path, 'rb') as img:
            message.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=os.path.basename(photo_path))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(message)
    server.quit()
