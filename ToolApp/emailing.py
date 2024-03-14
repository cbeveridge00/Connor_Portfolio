import smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Sends an email through gmail
# filename is the vectorized image file
# fullImage is (filename of) the colored image of all tools snapshotted
# Both images are attached to the document
def sendEmail(filename, fullImage, custname, batchname, email, phone):
    subject = "Tool Session from " + custname
    # body = "Batch " + batchname + ". /n" + email + "/n" + phone
    body = 'Customer Name: ' + custname + '\nBatch Name: ' + batchname + '\n' + email + '\n' + phone
    sender_email = "DXF-Submit@usfoamandetch.com"
    receiver_email = "DXF-Submit@usfoamandetch.com"
    password = "ctgu rjbs wvla kvjp"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # filename = "b.svg"  # In same directory as script

    # Open file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Open file in binary mode
    with open(fullImage, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part2 = MIMEBase("application", "octet-stream")
        part2.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part2)

    # Add header as key/value pair to attachment part
    part2.add_header(
        "Content-Disposition",
        f"attachment; filename= {fullImage}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    message.attach(part2)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)