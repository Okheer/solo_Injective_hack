import os
import asyncio
from dotenv import load_dotenv
from pyinjective.async_client_v2 import AsyncClient
from pyinjective.core.network import Network
from pyinjective.wallet import PrivateKey

load_dotenv()

class InjectiveClient:
    def __init__(self):
        self.network_name = os.getenv("INJECTIVE_NETWORK", "testnet")
        self.network = Network.testnet() if self.network_name == "testnet" else Network.mainnet()
        self.private_key_hex = os.getenv("INJ_PRIVATE_KEY")
        
        if self.private_key_hex:
            self.priv_key = PrivateKey.from_hex(self.private_key_hex)
            self.pub_key = self.priv_key.to_public_key()
            self.address = self.pub_key.to_address()
        else:
            self.priv_key = None
            self.address = None

    async def get_client(self) -> AsyncClient:
        return AsyncClient(self.network)

    async def get_account_portfolio(self):
        client = await self.get_client()
        if not self.address:
            return None
        return await client.get_account_portfolio(self.address.to_acc_bech32())

    async def place_derivative_order(
        self,
        market_id: str,
        price: float,
        quantity: float,
        leverage: float,
        is_buy: bool,
        is_market: bool = True
    ):
        """
        Place a derivative order on Injective.
        Builds, signs, and broadcasts a MsgCreateDerivativeMarketOrder or MsgCreateDerivativeLimitOrder to the chain.
        """
        if not self.priv_key:
            raise ValueError("No private key configured (INJ_PRIVATE_KEY). Cannot sign or broadcast on-chain transaction.")
            
        client = await self.get_client()

        subaccount_id = self.address.to_subaccount_id(index=0)

        from pyinjective.composer import Composer as GrpcComposer
        composer = GrpcComposer(network=self.network.string())
        
        margin = (price * quantity) / leverage
        order_type = 1 if is_buy else 2  
        
        # Build market order message
        msg = composer.MsgCreateDerivativeMarketOrder(
            sender=self.address.to_acc_bech32(),
            market_id=market_id,
            subaccount_id=subaccount_id,
            fee_recipient=self.address.to_acc_bech32(),
            price=str(price),
            quantity=str(quantity),
            margin=str(margin),
            order_type=order_type
        )
        
        # Fetch account details for signing sequence
        account_num, sequence = await client.get_account_sequence(self.address.to_acc_bech32())
        
        # Build Transaction
        from pyinjective.transaction import Transaction
        tx = Transaction(
            msgs=[msg],
            sequence=sequence,
            account_num=account_num,
            chain_id=self.network.chain_id,
        )
        
        # Sign transaction
        sim_sign_doc = tx.get_sign_doc(self.pub_key)
        signature = self.priv_key.sign(sim_sign_doc)
        tx.append_signature(signature, self.pub_key)
        
        # Broadcast transaction raw bytes
        tx_raw_bytes = tx.get_tx_bytes()
        res = await client.broadcast_tx_sync(tx_raw_bytes)
        return res
