"""
온톨로지 그래프 시각화 및 관계 확인
rdflib 기반으로 ontology.ttl 로드 & 쿼리
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL
from rdflib.plugins.sparql import prepareQuery
from pathlib import Path


ONTOLOGY_PATH = Path(__file__).parent / "ontology.ttl"
NS = Namespace("http://samantha.ai/ontology#")


def load_graph() -> Graph:
    g = Graph()
    g.parse(str(ONTOLOGY_PATH), format="turtle")
    print(f"트리플 수: {len(g)}")
    return g


def query_classes(g: Graph) -> list[str]:
    """온톨로지의 모든 클래스 조회"""
    q = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?class ?label WHERE {
        ?class a owl:Class .
        OPTIONAL { ?class rdfs:label ?label }
    }
    """
    results = g.query(q)
    classes = [str(row.class_).split("#")[-1] for row in results]
    return sorted(classes)


def query_properties(g: Graph, class_name: str) -> list[dict]:
    """특정 클래스의 속성 조회"""
    q = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://samantha.ai/ontology#>
    SELECT ?prop ?range WHERE {{
        {{ ?prop a owl:DatatypeProperty . }}
        UNION
        {{ ?prop a owl:ObjectProperty . }}
        ?prop rdfs:domain :{class_name} .
        OPTIONAL {{ ?prop rdfs:range ?range }}
    }}
    """
    results = g.query(q)
    props = []
    for row in results:
        props.append({
            "property": str(row.prop).split("#")[-1],
            "range": str(row.range).split("#")[-1] if row.range else "unknown"
        })
    return props


def print_ontology_summary(g: Graph):
    """온톨로지 구조 요약 출력"""
    print("\n" + "="*60)
    print("  SAMANTHA 온톨로지 구조")
    print("="*60)

    classes = query_classes(g)
    print(f"\n[클래스 목록] ({len(classes)}개)")
    for cls in classes:
        print(f"  • {cls}")

    key_classes = ["User", "Agent", "Relationship", "Memory", "Emotion", "Turn", "Topic", "Event"]
    for cls in key_classes:
        props = query_properties(g, cls)
        if props:
            print(f"\n[{cls}] 속성:")
            for p in props:
                print(f"  - {p['property']:30s} → {p['range']}")


def visualize_graph(g: Graph, output: str = "ontology_graph.html"):
    """pyvis로 인터랙티브 그래프 생성"""
    try:
        from pyvis.network import Network
    except ImportError:
        print("pip install pyvis 필요")
        return

    net = Network(height="800px", width="100%", directed=True)
    net.barnes_hut()

    color_map = {
        "Agent": "#FF6B6B",
        "User": "#4ECDC4",
        "Relationship": "#FFE66D",
        "Memory": "#A8E6CF",
        "Emotion": "#FF8B94",
        "Conversation": "#C3B1E1",
        "Turn": "#C3B1E1",
        "Topic": "#FFDAC1",
        "Event": "#B5EAD7",
        "Persona": "#FF6B6B",
        "Goal": "#E2F0CB",
    }

    added_nodes = set()

    for subj, pred, obj in g:
        subj_label = str(subj).split("#")[-1]
        pred_label = str(pred).split("#")[-1]
        obj_label  = str(obj).split("#")[-1]

        if pred_label in ("type", "subClassOf", "domain", "range"):
            continue

        for label in [subj_label, obj_label]:
            if label not in added_nodes:
                color = color_map.get(label, "#D3D3D3")
                net.add_node(label, label=label, color=color, size=20)
                added_nodes.add(label)

        net.add_edge(subj_label, obj_label, label=pred_label, arrows="to")

    net.save_graph(output)
    print(f"그래프 저장: {output}")


if __name__ == "__main__":
    g = load_graph()
    print_ontology_summary(g)
