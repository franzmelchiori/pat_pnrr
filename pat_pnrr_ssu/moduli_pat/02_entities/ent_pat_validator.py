from pathlib import Path
import xmlschema
from pyschematron import validate_document

XSD_FOR_VALIDATION = 'ent_pat_edilizia_persona_v1.0.0.xsd'
SCH_FOR_VALIDATION = 'ent_pat_edilizia_persona_v1.0.0.sch'
XML_TO_VALIDATE = 'ent_pat_edilizia_persona_v1.0.0.xml'


# XSD validation
xsd_file = open(XSD_FOR_VALIDATION)
xml_file = open(XML_TO_VALIDATE)
schema = xmlschema.XMLSchema(xsd_file)
if schema.is_valid(xml_file):
    print('XML valido')
else:
    schema.validate(xml_file)

# SCH validation
# result = validate_document(Path(XML_TO_VALIDATE), Path(SCH_FOR_VALIDATION))
# svrl = result.get_svrl()
# is_valid = result.is_valid()
