"""Financial Knowledge Graph Ontology with NetworkX."""

from typing import Any

import networkx as nx


class FinancialKnowledgeGraph:
    """Represents corporate, sector, peer, segment, and macro supply-chain relationships."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_default_graph()

    def _build_default_graph(self):
        # 1. Company Nodes
        companies = [
            ("TATAMOTORS.NS", {"name": "Tata Motors", "type": "Company", "market_cap_cr": 345200}),
            ("M&M.NS", {"name": "Mahindra & Mahindra", "type": "Company", "market_cap_cr": 320000}),
            ("MARUTI.NS", {"name": "Maruti Suzuki", "type": "Company", "market_cap_cr": 380000}),
            ("ASHOKLEY.NS", {"name": "Ashok Leyland", "type": "Company", "market_cap_cr": 65000}),
            ("BAJAJ-AUTO.NS", {"name": "Bajaj Auto", "type": "Company", "market_cap_cr": 250000}),
        ]
        for node_id, data in companies:
            self.graph.add_node(node_id, **data)

        # 2. Sector & Industry Nodes
        sectors = [
            ("Auto_Sector", {"name": "Automotive Sector", "type": "Sector"}),
            ("CV_Industry", {"name": "Commercial Vehicles", "type": "Industry"}),
            ("PV_Industry", {"name": "Passenger Vehicles", "type": "Industry"}),
            ("EV_Industry", {"name": "Electric Vehicles", "type": "Industry"}),
        ]
        for node_id, data in sectors:
            self.graph.add_node(node_id, **data)

        # 3. Segments for Tata Motors
        segments = [
            (
                "JLR_Segment",
                {"name": "Jaguar Land Rover (JLR)", "type": "Segment", "rev_share": "68.5%"},
            ),
            (
                "Tata_CV_Segment",
                {"name": "Tata Commercial Vehicles", "type": "Segment", "rev_share": "18.2%"},
            ),
            (
                "Tata_PV_Segment",
                {"name": "Tata Passenger & EV", "type": "Segment", "rev_share": "13.3%"},
            ),
        ]
        for node_id, data in segments:
            self.graph.add_node(node_id, **data)

        # 4. Macro & Supply Chain Factors
        macro = [
            ("Macro_Steel", {"name": "Steel & Aluminum Commodity Prices", "type": "Macro"}),
            ("Macro_Semiconductor", {"name": "Automotive Semiconductor Supply", "type": "Macro"}),
            ("Macro_InterestRate", {"name": "RBI & Global Interest Rates", "type": "Macro"}),
            ("Macro_BatteryLithium", {"name": "Lithium-Ion Battery Cell Cost", "type": "Macro"}),
        ]
        for node_id, data in macro:
            self.graph.add_node(node_id, **data)

        # Edges
        # Belongs to
        for ticker in ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "ASHOKLEY.NS", "BAJAJ-AUTO.NS"]:
            self.graph.add_edge(ticker, "Auto_Sector", relation="belongs_to_sector")

        # Segments
        self.graph.add_edge("JLR_Segment", "TATAMOTORS.NS", relation="subsidiary_of")
        self.graph.add_edge("Tata_CV_Segment", "TATAMOTORS.NS", relation="business_unit_of")
        self.graph.add_edge("Tata_PV_Segment", "TATAMOTORS.NS", relation="business_unit_of")

        # Competitions
        self.graph.add_edge("TATAMOTORS.NS", "M&M.NS", relation="competes_with", segment="SUV & EV")
        self.graph.add_edge(
            "TATAMOTORS.NS", "MARUTI.NS", relation="competes_with", segment="Passenger Cars"
        )
        self.graph.add_edge(
            "TATAMOTORS.NS",
            "ASHOKLEY.NS",
            relation="competes_with",
            segment="Heavy Commercial Vehicles",
        )

        # Macro Impacts
        self.graph.add_edge(
            "Macro_Steel", "TATAMOTORS.NS", relation="input_cost_impact", severity="Medium"
        )
        self.graph.add_edge(
            "Macro_Semiconductor", "JLR_Segment", relation="supply_bottleneck", severity="High"
        )
        self.graph.add_edge(
            "Macro_InterestRate",
            "Tata_CV_Segment",
            relation="fleet_financing_cost",
            severity="High",
        )
        self.graph.add_edge(
            "Macro_BatteryLithium",
            "Tata_PV_Segment",
            relation="ev_margin_impact",
            severity="Medium",
        )

    def get_company_peers(self, ticker: str = "TATAMOTORS.NS") -> list[dict[str, Any]]:
        """Finds competitor nodes in the knowledge graph."""
        peers = []
        if ticker in self.graph:
            for neighbor in self.graph.neighbors(ticker):
                edge_data = self.graph.get_edge_data(ticker, neighbor)
                if edge_data.get("relation") == "competes_with":
                    node_data = self.graph.nodes[neighbor]
                    peers.append(
                        {
                            "ticker": neighbor,
                            "name": node_data.get("name", neighbor),
                            "competing_segment": edge_data.get("segment", "General"),
                        }
                    )
        return peers

    def get_risk_factors(self, ticker: str = "TATAMOTORS.NS") -> list[dict[str, Any]]:
        """Identifies upstream macro dependencies."""
        risks = []
        for u, v, data in self.graph.edges(data=True):
            if v == ticker or (
                v in ["JLR_Segment", "Tata_CV_Segment", "Tata_PV_Segment"]
                and "TATAMOTORS" in ticker
            ):
                if data.get("relation") in [
                    "input_cost_impact",
                    "supply_bottleneck",
                    "fleet_financing_cost",
                    "ev_margin_impact",
                ]:
                    macro_node = self.graph.nodes[u]
                    risks.append(
                        {
                            "factor": macro_node.get("name", u),
                            "relationship": data.get("relation"),
                            "impact_severity": data.get("severity", "Medium"),
                        }
                    )
        return risks

    def export_graph_json(self) -> dict[str, Any]:
        """Exports nodes and links formatted for Plotly / D3 / UI graph rendering."""
        nodes = []
        for n, d in self.graph.nodes(data=True):
            nodes.append({"id": n, "label": d.get("name", n), "type": d.get("type", "Unknown")})

        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, "relation": d.get("relation", "")})

        return {"nodes": nodes, "edges": edges}
