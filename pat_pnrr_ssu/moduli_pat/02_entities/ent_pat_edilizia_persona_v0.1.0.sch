<?xml version="1.0" encoding="UTF-8"?>
<sch:schema
    xmlns:sch="http://purl.oclc.org/dsdl/schematron"
    queryBinding="xslt2">
    
    <sch:ns uri="../02_entities/ent_pat_edilizia_persona_v0.1.0.xsd" prefix="epepers"/>
    
    <sch:pattern id="persona_ab" abstract="true">
        <sch:rule id="rule_persona"  context="$entita_pat_edilizia_persona">
            <sch:let name="keysStoricoComuni" value="document('../01_vocabularies/voc_comuni_italiani_storico.xml')//Row"/> 
            <sch:let name="keysProvince" value="document('../01_vocabularies/voc_province_italiane_v1.0.0.xml')//Row"/> 
            <sch:let name="keysStati" value="document('../01_vocabularies/voc_stati_v1.0.0.xml')//Row"/> 

            <sch:let name="comune_nascita" value="normalize-space(epepers:comuneNascita)"/>
            <sch:let name="provincia_nascita" value="normalize-space(epepers:provinciaNascita)"/>
            <sch:let name="stato_nascita" value="normalize-space(epepers:statoNascita)"/>
            
            <sch:assert
                id="persona_ab-ass_comune_nascita_check"
                test="$comune_nascita='' or 
                     count($keysComuni[normalize-space(Value[@ColumnRef='denominazione']/SimpleValue) = $comune_nascita]) = 1">
                Comune di nascita non esiste (<sch:value-of select="$comune_nascita"/>)
            </sch:assert>
            
            <sch:assert
                id="persona_ab-ass_provincia_nascita_check"
                test="$provincia_nascita='' or 
                     count($keysProvincia[normalize-space(Value[@ColumnRef='sigla']/SimpleValue) = $provincia_nascita]) = 1">
                Provincia di nascita non esiste (<sch:value-of select="$provincia_nascita"/>)
            </sch:assert>
            
            <sch:assert
                id="persona_ab-ass_stato_nascita_check"
                test="$stato_nascita='' or 
                     count($keysStati[normalize-space(Value[@ColumnRef='denominazione']/SimpleValue) = $stato_nascita]) = 1">
                Stato di nascita non esiste (<sch:value-of select="$stato_nascita"/>)
            </sch:assert>
        </sch:rule>
    </sch:pattern>

    <sch:pattern id="entita_pat_edilizia_persona" abstract="false" is-a="persona_ab">
        <sch:param name="entita_pat_edilizia_persona" value="epepers:entita_pat_edilizia_persona"/>
    </sch:pattern>
</sch:schema>
