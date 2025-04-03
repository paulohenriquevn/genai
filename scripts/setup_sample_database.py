import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

def create_sample_database(db_path='database.db'):
    """Cria um banco de dados SQLite de exemplo para testes do GenBI"""
    
    # Conectar ao banco de dados
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Criar tabela de clientes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        city TEXT,
        country TEXT,
        created_at DATETIME
    )
    ''')
    
    # Criar tabela de produtos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT,
        category TEXT,
        price DECIMAL(10,2),
        cost DECIMAL(10,2)
    )
    ''')
    
    # Criar tabela de pedidos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date DATE,
        status TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
    )
    ''')
    
    # Criar tabela de itens do pedido
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        price DECIMAL(10,2),
        FOREIGN KEY(order_id) REFERENCES orders(order_id),
        FOREIGN KEY(product_id) REFERENCES products(product_id)
    )
    ''')
    
    # Gerar dados de exemplo
    
    # Clientes
    customers_data = [
        (1, 'João Silva', 'joao@email.com', 'São Paulo', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (2, 'Maria Santos', 'maria@email.com', 'Rio de Janeiro', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (3, 'Pedro Oliveira', 'pedro@email.com', 'Belo Horizonte', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (4, 'Ana Costa', 'ana@email.com', 'Salvador', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (5, 'Carlos Souza', 'carlos@email.com', 'Curitiba', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365)))
    ]
    cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?)', customers_data)
    
    # Produtos
    products_data = [
        (1, 'Notebook Dell', 'Eletrônicos', 4500.00, 3200.00),
        (2, 'Smartphone Samsung', 'Eletrônicos', 2800.00, 1900.00),
        (3, 'Tablet Apple', 'Eletrônicos', 3200.00, 2100.00),
        (4, 'Monitor LG', 'Eletrônicos', 1200.00, 800.00),
        (5, 'Teclado Mecânico', 'Acessórios', 350.00, 180.00),
        (6, 'Mouse Gamer', 'Acessórios', 250.00, 120.00),
        (7, 'Cadeira Gamer', 'Móveis', 1800.00, 1200.00),
        (8, 'Webcam HD', 'Eletrônicos', 450.00, 250.00)
    ]
    cursor.executemany('INSERT OR REPLACE INTO products VALUES (?,?,?,?,?)', products_data)
    
    # Gerar pedidos e itens de pedido com dados aleatórios
    def generate_orders(num_orders=150):
        order_data = []
        order_items_data = []
        
        order_item_id_counter = 1
        
        for i in range(1, num_orders + 1):
            # Distribuir datas de pedido nos últimos 12 meses para ter dados históricos melhores
            # Mais pedidos recentes (60% nos últimos 3 meses, 40% nos 9 meses anteriores)
            if random.random() < 0.6:
                # Últimos 3 meses
                order_date = datetime.now() - timedelta(days=random.randint(0, 90))
            else:
                # Entre 3 e 12 meses atrás
                order_date = datetime.now() - timedelta(days=random.randint(91, 365))
            
            # Escolher cliente aleatório
            customer_id = random.choice([c[0] for c in customers_data])
            
            # Status possíveis (mais pedidos entregues do que pendentes)
            status_weights = {'Entregue': 0.6, 'Enviado': 0.2, 'Processando': 0.1, 'Pendente': 0.1}
            status_options = list(status_weights.keys())
            status_probabilities = list(status_weights.values())
            status = random.choices(status_options, weights=status_probabilities, k=1)[0]
            
            order_data.append((i, customer_id, order_date.date(), status))
            
            # Gerar itens do pedido (média maior para mais dados)
            num_items = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.3, 0.1], k=1)[0]
            
            # Produtos mais vendidos têm maior probabilidade
            product_weights = {1: 0.3, 2: 0.25, 3: 0.15, 4: 0.1, 5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05}
            
            # Selecionar produtos com repetições possíveis
            products_in_order = []
            for _ in range(num_items):
                product_id = random.choices(
                    list(product_weights.keys()),
                    weights=list(product_weights.values()),
                    k=1
                )[0]
                product = next((p for p in products_data if p[0] == product_id), None)
                if product:
                    products_in_order.append(product)
            
            # Garantir que temos pelo menos um produto no pedido
            if not products_in_order and products_data:
                products_in_order = [random.choice(products_data)]
            
            for product in products_in_order:
                # Quantidades diferentes (mais comum 1 ou 2)
                quantity = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05], k=1)[0]
                
                # Preço pode variar um pouco (+/- 10% do preço base)
                price_variation = random.uniform(0.9, 1.1)
                base_price = float(product[3])  # Índice 3 contém o preço no products_data
                price = round(base_price * price_variation, 2)
                
                order_items_data.append((
                    order_item_id_counter,  # order_item_id único
                    i,  # order_id
                    product[0],  # product_id
                    quantity,
                    price  # preço do produto com variação
                ))
                order_item_id_counter += 1
        
        return order_data, order_items_data
    
    # Gerar mais produtos para diversidade
    additional_products = [
        (9, 'Fone de Ouvido Bluetooth', 'Eletrônicos', 180.00, 100.00),
        (10, 'Teclado Sem Fio', 'Acessórios', 120.00, 70.00),
        (11, 'Mouse Óptico', 'Acessórios', 65.00, 30.00),
        (12, 'Webcam HD Pro', 'Eletrônicos', 350.00, 180.00),
        (13, 'Headset Gamer', 'Acessórios', 290.00, 150.00),
        (14, 'SSD 500GB', 'Eletrônicos', 450.00, 280.00),
        (15, 'Monitor Ultrawide', 'Eletrônicos', 2100.00, 1400.00),
        (16, 'Mesa de Escritório', 'Móveis', 580.00, 300.00),
        (17, 'Cadeira Ergonômica', 'Móveis', 1200.00, 750.00),
        (18, 'Suporte para Notebook', 'Acessórios', 150.00, 80.00)
    ]
    
    # Adicionar novos produtos
    products_data.extend(additional_products)
    cursor.executemany('INSERT OR REPLACE INTO products VALUES (?,?,?,?,?)', additional_products)
    
    # Adicionar mais clientes para diversidade
    additional_customers = [
        (6, 'Roberto Almeida', 'roberto@email.com', 'Recife', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (7, 'Amanda Silva', 'amanda@email.com', 'Fortaleza', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (8, 'Lucas Pereira', 'lucas@email.com', 'Brasília', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (9, 'Juliana Costa', 'juliana@email.com', 'Manaus', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365))),
        (10, 'Rodrigo Santos', 'rodrigo@email.com', 'Porto Alegre', 'Brasil', datetime.now() - timedelta(days=random.randint(30, 365)))
    ]
    
    # Adicionar novos clientes
    customers_data.extend(additional_customers)
    cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?)', additional_customers)
    
    # Gerar e inserir pedidos (150 pedidos para ter mais dados)
    orders, order_items = generate_orders(150)
    cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?,?,?,?)', orders)
    cursor.executemany('INSERT OR REPLACE INTO order_items VALUES (?,?,?,?,?)', order_items)
    
    # Commitar e fechar conexão
    conn.commit()
    conn.close()
    
    print(f"Banco de dados de exemplo criado em {db_path}")

if __name__ == "__main__":
    create_sample_database()