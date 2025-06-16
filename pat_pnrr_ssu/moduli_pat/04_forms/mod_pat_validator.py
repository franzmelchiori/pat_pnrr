import xmlschema

XSD_FOR_VALIDATION = 'mod_pat_edilizia_comunicazione_inizio_lavori_v0.1.0.xsd'
XML_TO_VALIDATE = 'mod_pat_edilizia_comunicazione_inizio_lavori_v0.1.0.xml'


schema_file = open(XSD_FOR_VALIDATION)
schema = xmlschema.XMLSchema(schema_file)

xml_file = open(XML_TO_VALIDATE)
if schema.is_valid(xml_file):
    print('XML valido')
else:
    schema.validate(xml_file)
