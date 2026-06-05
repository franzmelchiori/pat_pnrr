"""
    PAT SUE: XML parser
    ⌘ 2026 Francesco Melchiori
"""


# import xmlschema
from lxml import etree


# XSD_FOR_VALIDATION = 'mod_pat_comunicazione_inizio_lavori_v1.0.0.xsd'
# XML_TO_VALIDATE = 'mod_pat_comunicazione_inizio_lavori_v1.0.0.xml'

XML_PATH = '/home/franzmelchiori/ds220melchiori/franzmelchiori/projects/pat_pnrr/' + \
           'pat_pnrr_ssu/moduli_pat/04_forms/' + \
           'mod_pat_comunicazione_inizio_lavori_v1.0.0.xml'
XSD_PATH = '/home/franzmelchiori/ds220melchiori/franzmelchiori/projects/pat_pnrr/' + \
           'pat_pnrr_ssu/moduli_pat/04_forms/' + \
           'mod_pat_comunicazione_inizio_lavori_v1.0.0.xsd'


def prettyprint(element, **kwargs):
    xml = etree.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end='')


if __name__ == '__main__':


    # schema_file = open(XSD_FOR_VALIDATION)
    # schema = xmlschema.XMLSchema(schema_file)

    # xml_file = open(XML_TO_VALIDATE)
    # if schema.is_valid(xml_file):
    #     print('XML valido')
    # else:
    #     schema.validate(xml_file)


    ## parse XML from XML file
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(XML_PATH, parser)
    root = tree.getroot()

    for element in root.iter("*"):
        if element.text is not None and not element.text.strip():
            element.text = None
        elif element.text is not None and element.text.strip():
            element.text = element.text.strip('\n')
            element.text = element.text.strip(' ')
        else:
            pass

    ## print XML
    if False:
        print(etree.tostring(tree))
        prettyprint(tree)
    
    ## browse XML
    if False:
        for element in root.iter():
            print(f'{element.tag} : {element.text}')
    
    ## browse XML with XPath
    if False:
        scia = tree.xpath('/scia:ModuloSCIA', namespaces={'scia': 'http://www.agid.gov.it/MODULI_PA/SUE/modulo_scia'})
        print(scia[0][0][0].tag)
        print(scia[0][0][0].text)
        titolare = tree.xpath('/scia:ModuloSCIA/scia:schedaTitolare', namespaces={'scia': 'http://www.agid.gov.it/MODULI_PA/SUE/modulo_scia'})
        print(titolare[0][0].tag)
        print(titolare[0][0].text)

        find = tree.find('.//scia:schedaTitolare', namespaces={'scia': 'http://www.agid.gov.it/MODULI_PA/SUE/modulo_scia'})
        print(tree.getpath(find))
        find = tree.find('.//suecommon:riferimentiCatastali', namespaces={'suecommon': 'http://www.agid.gov.it/MODULI_PA/SUE/sue_common_components'})
        print(tree.getpath(find))

        find = etree.XPath('.//scia:schedaTitolare', namespaces={'scia': 'http://www.agid.gov.it/MODULI_PA/SUE/modulo_scia'})
        print(find(tree)[0].tag)
        find = etree.XPath('.//suecommon:riferimentiCatastali', namespaces={'suecommon': 'http://www.agid.gov.it/MODULI_PA/SUE/sue_common_components'})
        print(find(tree)[0].tag)

        count_elements = etree.XPath('count(.//*[local-name() = $name])')
        print(count_elements(tree, name = 'cognome'))

    ## parse XSD from XSD files
    xsd = etree.parse(XSD_PATH)
    schema = etree.XMLSchema(xsd)

    ## validate XML with XSD
    if schema.validate(tree):
        print('XML is valid\n')
    else:
        print('XML is NOT valid\n')
        for error in schema.error_log:
            print(str(error) + '\n')

    ## assert XML with XSD
    # schema.assertValid(tree)

    ## valuta errori di validazione tra XML PAT e XSD IT, es.:
    ##     - elementi XML PAT per cui non esiste il relativo elemento XSD IT (dati aggiunti per PAT)
    ##     - elementi XML PAT non alimentati per il relativo elemento XSD IT (dati inesistenti in PAT)
    ##     - elementi XML PAT non validi (ma tipizzati) per il relativo elemento XSD IT
    ##     - elementi XML PAT mancanti (ma obbligatori) per il relativo elemento XSD IT
