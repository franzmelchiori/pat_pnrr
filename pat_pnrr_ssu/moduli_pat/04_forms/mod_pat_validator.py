import xmlschema


schema_file = open('mod_pat_edilizia_comunicazione_inizio_lavori_v0.1.0.xsd')
schema = xmlschema.XMLSchema(schema_file)

xml_file = open('mod_pat_edilizia_comunicazione_inizio_lavori_v0.1.0.xml')
if schema.is_valid(xml_file):
    print('XML valido')
schema.validate(xml_file)
