import re

camelSplitter = re.compile(r'_|\W+|\s+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9]+)|(?<=[0-9])(?=[a-zA-Z]+)')
def camelSplit(string):
    '''Split camel case words. E.g.,

    >>> camelSplit("HelloWorld.java {{RSSOwl_AtomFeedLoader}}")
    ['Hello', 'World', 'java', 'RSS', 'Owl', 'Atom', 'Feed', 'Loader']
    '''
    result = []
    last = 0
    for match in camelSplitter.finditer(string):
        if string[last:match.start()] != '':
            result.append(string[last:match.start()])
        last = match.end()
    if string[last:] != '':
        result.append(string[last:])
    return result

fix_regex = re.compile(r'[\\/]+')
def fix(string):
    return fix_regex.sub('/',string)

normalize_eclipse = re.compile(r"L([^;]+);.*")
normalize_path = re.compile(r".*src\/(.*)\.java")
def normalize(string):
    '''
    Return the class indicated in the string.
    File-name example:
    Raw file name: jEdit/src/org/gjt/sp/jedit/gui/StatusBar.java
    Normalized file name: org/gjt/sp/jedit/gui/StatusBar

    '''
    m = normalize_eclipse.match(string)
    if m:
        return m.group(1)
    n = normalize_path.match(fix(string))
    if n:
        return n.group(1)
    return ''


package_regex = re.compile(r"(.*)/[a-zA-Z0-9]+")
def package(string):
    '''Return the package.'''
    m = package_regex.match(normalize(string))
    if m:
        return m.group(1)
    return ''

project_regex = re.compile(r"\/(.*)\/src/.*")
def project(string):
    '''Return the project.'''
    m = project_regex.match(fix(string))
    if m:
        return m.group(1)
    return ''


"""Store / Load Functions"""
def storeGraphs(graphs):
    f = open('graphs.pk1', 'wb')
    cPickle.dump(graphs, f)
    f.close()


def storeNavigation(nav):
    f = open('nav.pk1', 'wb')
    cPickle.dump(nav, f)
    f.close()


def storeTopology(top):
    f = open('topology.pkl', 'wb')
    cPickle.dump(top, f)
    f.close()


def storeScent(scent):
    f = open('scent.pkl', 'wb')
    cPickle.dump(scent, f)
    f.close()

def loadGraphs():
    f = open('graphs.pk1', 'rb')
    graphs = cPickle.load(f)
    f.close()
    return graphs


def loadNav():
    f = open('nav.pk1', 'rb')
    nav = cPickle.load(f)
    f.close()
    return nav


def loadTopologyPickle():
    f = open('topology.pkl', 'rb')
    topology = cPickle.load(f)
    f.close()
    return topology


def loadScentPickle():
    f = open('scent.pkl', 'rb')
    scent = cPickle.load(f)
    f.close()
    return scent
