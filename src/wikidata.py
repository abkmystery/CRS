import requests
import time


class WikidataFetcher:
    def __init__(self):
        self.endpoint = "https://www.wikidata.org/w/api.php"
        self.headers = {'User-Agent': 'CRS_Bot/1.0 (Commercial_Safe_Open_Source)'}

        # Simple cache for property labels to make answers readable
        self.prop_map = {
            'P31': 'instance of',
            'P279': 'subclass of',
            'P106': 'occupation',
            'P39': 'position held',
            'P569': 'date of birth',
            'P2048': 'height',
            'P2218': 'net worth',
            'P856': 'website',
            'P17': 'country'
        }

    def search_entity(self, label):
        """Finds the Wikidata ID (Q-ID)"""
        params = {
            'action': 'wbsearchentities',
            'search': label,
            'language': 'en',
            'format': 'json',
            'limit': 1
        }
        try:
            r = requests.get(self.endpoint, params=params, headers=self.headers)
            data = r.json()
            if data.get('search'):
                return data['search'][0]['id']
        except Exception as e:
            print(f"   ⚠️ Wiki Error: {e}")
        return None

    def get_details(self, qid):
        """Fetches Description + Claims"""
        if not qid: return None, [], []

        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'props': 'descriptions|claims|labels',
            'languages': 'en',
            'format': 'json'
        }

        props_out = []
        rels_out = []
        description = "No description available."
        label = qid

        try:
            r = requests.get(self.endpoint, params=params, headers=self.headers)
            data = r.json()
            entity = data.get('entities', {}).get(qid, {})

            # 1. Get Real Description
            desc_obj = entity.get('descriptions', {}).get('en', {})
            if desc_obj:
                description = desc_obj.get('value', description)

            # 2. Get Real Label
            lbl_obj = entity.get('labels', {}).get('en', {})
            if lbl_obj:
                label = lbl_obj.get('value', label)

            # 3. Get Properties
            claims = entity.get('claims', {})

            for prop_id, items in claims.items():
                # Map P codes to text (e.g. P31 -> instance of)
                human_key = self.prop_map.get(prop_id, prop_id)

                for item in items[:2]:  # Limit to top 2 values per property
                    mainsnak = item.get('mainsnak', {})
                    datavalue = mainsnak.get('datavalue', {})
                    dtype = mainsnak.get('datatype', '')

                    # Handle Strings/Amounts
                    if dtype in ['string', 'quantity']:
                        val = datavalue.get('value')
                        if isinstance(val, dict): val = val.get('amount', str(val))
                        props_out.append({'key': human_key, 'value': str(val).replace('+', '')})

                    # Handle Relations (Wiki Items)
                    elif dtype == 'wikibase-item':
                        target_id = datavalue.get('value', {}).get('id')
                        if target_id:
                            rels_out.append({'type': human_key, 'target_id': f"wiki_{target_id}", 'source': 'wikidata'})

        except Exception as e:
            pass

        return description, label, props_out, rels_out