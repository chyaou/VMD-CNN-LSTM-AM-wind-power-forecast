from twilio.rest import Client


def send_message(messages, receive_number='+86xxxxx'):
    """
    | *信息内容* | *接收信息号码* |
    | 自动发送 | +86171XXXX1121 |
    :param messages: 发送信息的内容
    :receive_number: 需要再twilio网站验证号码才能接收
    网址:https://www.twilio.com/console/phone-numbers/verified
    """
    phone_number = '+18586486603'  # 步骤6由网站分配的
    account_sid = account_sid
    auth_token = auth_token

    def beging_sending_message(msg, target_number):
        try:
            client = Client(account_sid, auth_token)
            client.messages.create(to=target_number, from_=phone_number, body=msg)
            return True
        except Exception:
            return False

    if beging_sending_message(messages, receive_number):
        print("短信已成功发送至%s" % receive_number)
    else:
        print("短信发送失败!!!")
