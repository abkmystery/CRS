import os
import flatbuffers
import nltk
from nltk.corpus import wordnet as wn
import sys

sys.path.append('./generated')
import crs.Concept as C
import crs.Embedding as E
import crs.Relation as R
import crs.Property as P
import crs.Evidence as Ev  # <--- NEW: Import Evidence

try:
    wn.all_synsets()
except LookupError:
    nltk.download('wordnet'); nltk.download('omw-1.4')


class ConceptBuilder:
    def __init__(self, output_dir="data/concepts"):
        self.builder = flatbuffers.Builder(1024)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _create_string(self, s):
        return self.builder.CreateString(s) if s else None

    def _create_vector(self, data, start_fn, pack_fn=None):
        if not data: return None
        if isinstance(data[0], str):
            offsets = [self.builder.CreateString(x) for x in data]
            start_fn(self.builder, len(offsets))
            for o in reversed(offsets): self.builder.PrependUOffsetTRelative(o)
            return self.builder.EndVector()
        else:
            start_fn(self.builder, len(data))
            for x in reversed(data):
                if pack_fn:
                    pack_fn(x)
                else:
                    self.builder.PrependFloat32(x)
            return self.builder.EndVector()

    def _create_embedding(self, vector_data):
        if not vector_data: return None
        vec_offset = self._create_vector(vector_data, E.StartVectorVector)
        E.Start(self.builder)
        E.AddVector(self.builder, vec_offset)
        return E.End(self.builder)

    def _create_relations(self, rels):
        if not rels: return None
        offsets = []
        for r in rels:
            type_off = self._create_string(r['type'])
            tid_off = self._create_string(r['target_id'])
            src_off = self._create_string(r['source'])
            R.Start(self.builder)
            R.AddType(self.builder, type_off)
            R.AddTargetId(self.builder, tid_off)
            R.AddSource(self.builder, src_off)
            R.AddConfidence(self.builder, r.get('confidence', 1.0))
            offsets.append(R.End(self.builder))
        C.StartRelationsVector(self.builder, len(offsets))
        for o in reversed(offsets): self.builder.PrependUOffsetTRelative(o)
        return self.builder.EndVector()

    def _create_properties(self, props):
        if not props: return None
        offsets = []
        for p in props:
            k_off = self._create_string(p['key'])
            v_off = self._create_string(str(p['value']))
            P.Start(self.builder)
            P.AddKey(self.builder, k_off)
            P.AddValue(self.builder, v_off)
            offsets.append(P.End(self.builder))
        C.StartPropertiesVector(self.builder, len(offsets))
        for o in reversed(offsets): self.builder.PrependUOffsetTRelative(o)
        return self.builder.EndVector()

    # --- NEW: Evidence Builder ---
    def _create_evidence(self, evidence_list):
        if not evidence_list: return None
        offsets = []
        for e in evidence_list:
            src_off = self._create_string(e.get('source_type', 'unknown'))
            snip_off = self._create_string(e.get('snippet', ''))
            url_off = self._create_string(e.get('url', ''))

            Ev.Start(self.builder)
            Ev.AddSourceType(self.builder, src_off)
            Ev.AddSnippet(self.builder, snip_off)
            Ev.AddUrl(self.builder, url_off)
            Ev.AddConfidence(self.builder, e.get('confidence', 1.0))
            offsets.append(Ev.End(self.builder))

        C.StartEvidenceVector(self.builder, len(offsets))
        for o in reversed(offsets): self.builder.PrependUOffsetTRelative(o)
        return self.builder.EndVector()

    def build_concept(self, data):
        self.builder = flatbuffers.Builder(1024)

        id_off = self._create_string(data['id'])
        lbl_off = self._create_string(data['label'])
        def_off = self._create_string(data.get('definition', ''))

        # New: Aliases
        alias_off = self._create_vector(data.get('aliases', []), C.StartAliasesVector)

        txt_emb = self._create_embedding(data.get('text_embedding'))
        rels_off = self._create_relations(data.get('relations', []))
        props_off = self._create_properties(data.get('properties', []))
        ev_off = self._create_evidence(data.get('evidence', []))  # <--- NEW

        C.Start(self.builder)
        C.AddId(self.builder, id_off)
        C.AddLabel(self.builder, lbl_off)
        C.AddDefinition(self.builder, def_off)
        if alias_off: C.AddAliases(self.builder, alias_off)
        if txt_emb: C.AddTextEmbedding(self.builder, txt_emb)
        if rels_off: C.AddRelations(self.builder, rels_off)
        if props_off: C.AddProperties(self.builder, props_off)
        if ev_off: C.AddEvidence(self.builder, ev_off)  # <--- NEW

        concept = C.End(self.builder)
        self.builder.Finish(concept)

        buf = self.builder.Output()
        filename = os.path.join(self.output_dir, f"{data['id']}.bin")
        with open(filename, 'wb') as f:
            f.write(buf)
        return filename