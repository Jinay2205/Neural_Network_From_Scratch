import hashlib
import random

class Block:
    def __init__(self, index, previous_hash, timestamp, data, validator):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.validator = validator
        self.hash = self.calculate_hash()
        
    def calculate_hash(self):
        data_to_hash = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.validator}"
        return hashlib.sha256(data_to_hash.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.stakeholders = {}  # Dictionary to store stakeholder balances (stake)
    
    def create_genesis_block(self):
        return Block(0, "0", "2024-01-01 00:00:00", "Genesis Block", "Genesis Validator")
    
    def get_last_block(self):
        return self.chain[-1]
    
    def add_stake(self, validator, amount):
        if validator in self.stakeholders:
            self.stakeholders[validator] += amount
        else:
            self.stakeholders[validator] = amount

    def select_validator(self):
        total_stake = sum(self.stakeholders.values())
        selected = random.choices(
            list(self.stakeholders.keys()), 
            weights=list(self.stakeholders.values()), 
            k=1
        )[0]
        return selected

    def add_block(self, data, validator):
        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            previous_hash=last_block.hash,
            timestamp="2024-11-06 00:00:00",  # Placeholder timestamp
            data=data,
            validator=validator
        )
        if self.validate_block(new_block, validator):
            self.chain.append(new_block)
            print(f"Block {new_block.index} added by {validator}.")
        else:
            print("Block validation failed.")

    def validate_block(self, block, validator):
    # Check if the validator is an authorized stakeholder
        return validator in self.stakeholders

    def show_chain(self):
        for block in self.chain:
            print(f"Index: {block.index}, Hash: {block.hash}, Validator: {block.validator}, Data: {block.data}")

# Usage
blockchain = Blockchain()
blockchain.add_stake("Validator1", 50)
blockchain.add_stake("Validator2", 100)
blockchain.add_stake("Validator3", 200)

# Randomly select validator based on stake
for i in range(5):
    validator = blockchain.select_validator()
    blockchain.add_block(f"Block Data {i}", validator)

# Display the blockchain
blockchain.show_chain()
