import pika
import sys
import threading
import time

class IMessageQueue():

    is_synchronize = False
    reply_value = None

    def __init__(self, tag, publish_key='', queue_name='', receive_key='', callback_func='', exchange_type='topic'):
        self.cond = threading.Condition()

        self.tag = tag
        self.user_name = 'rabbitmq'
        self.user_pwd = 'rabbitmq@0'
        self.ip = '10.89.100.14'
        self.port = 5672

        self.queue_name = queue_name
        self.receive_key = receive_key
        self.publish_key = publish_key
        self.callback_func = callback_func
        self.exchange = 'nlp_'+exchange_type
        self.exchange_type = exchange_type

        crt = pika.PlainCredentials(self.user_name, self.user_pwd)

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.ip, self.port, '/', credentials=crt))
        self.chan = self.connection.channel()

        self.chan.exchange_declare(exchange=self.exchange, type=self.exchange_type)

        if self.receive_key and self.queue_name:
            def target():
                self.chan.basic_qos(prefetch_count=1)
                #result = self.chan.queue_declare(exclusive=True)   
                #self.queue_name = result.method.queue
                self.chan.queue_declare(queue=self.queue_name, auto_delete=True)
                self.chan.queue_bind(exchange=self.exchange,
                           queue=self.queue_name,
                           routing_key=self.receive_key)
                self.chan.basic_consume(self.callback, queue=self.queue_name)
                self.chan.start_consuming()
            t = threading.Thread(target=target)
            t.start()

        print("[%s] start" %(self.tag))

    def __exit__(self):
        self.connection.close()

    def publish(self, value, routing_key=''):
        self.reply_value = None
        key = self.publish_key
        if routing_key:
            key = routing_key
        print("[%s] pubs key %s, body %s" %(self.tag, key, value))
        if key:
            self.chan.basic_publish(exchange=self.exchange, 
                    routing_key=key,
                    body=value,
                    properties=pika.BasicProperties(
                        content_type = 'text/plain', message_id = self.tag,
                    ))

    def callback(self, ch, method, properties, body):
        key = method.routing_key
        body = body.decode('utf-8')
        print("[%s] recv exchange %s, key %s, body %s" %(self.tag, method.exchange, key, body))
        if self.callback_func:
            #print('call ', str(self.callback_func))
            self.callback_func(key, body, self.publish)
        #else:
        #    print('no callback found')
        self.chan.basic_ack(delivery_tag = method.delivery_tag)
        self.reply_value = body
        if self.is_synchronize:
            self.cond.acquire()
            self.cond.notify()
            self.cond.release()

    def request_synchronize(self, value):
        self.is_synchronize = True
        self.cond.acquire()
        self.publish(value)
        self.cond.wait()
        self.cond.release()
        return self.reply_value
