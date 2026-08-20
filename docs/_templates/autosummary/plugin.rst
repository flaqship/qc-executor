{{ name | escape | underline }}

.. module:: {{ fullname }}

{% set deps = plugin_dependencies.get(fullname) if plugin_dependencies is defined else none %}
{% if deps %}
Dependencies
------------
{% if deps.requires %}
Install with ``pip install qc-executor[{{ deps.extra }}]``. Requires:
{% for req in deps.requires %}
* ``{{ req }}``
{% endfor %}
{% else %}
Included in the core package — no additional dependencies required.
{% endif %}
{% endif %}

{% set ns = namespace(executors=[], circuits=[], operators=[], others=[]) %}
{% for item in classes %}
{% if item.endswith("Executor") %}{% set ns.executors = ns.executors + [item] %}
{% elif item.endswith("Circuit") %}{% set ns.circuits = ns.circuits + [item] %}
{% elif item.endswith("Operator") %}{% set ns.operators = ns.operators + [item] %}
{% else %}{% set ns.others = ns.others + [item] %}
{% endif %}
{% endfor %}

{% if ns.executors %}
Executor
--------
{% for cls in ns.executors %}
.. autoclass:: {{ cls }}
   :members:
   :show-inheritance:
   :member-order: bysource
{% endfor %}
{% endif %}

{% if ns.circuits or ns.operators %}
Native abstraction
------------------
{% if ns.circuits %}
Circuit
~~~~~~~
{% for cls in ns.circuits %}
.. autoclass:: {{ cls }}
   :members:
   :show-inheritance:
   :member-order: bysource
{% endfor %}
{% endif %}
{% if ns.operators %}
Operator
~~~~~~~~
{% for cls in ns.operators %}
.. autoclass:: {{ cls }}
   :members:
   :show-inheritance:
   :member-order: bysource
{% endfor %}
{% endif %}
{% endif %}

{% if ns.others %}
Other classes
-------------
{% for cls in ns.others %}
.. autoclass:: {{ cls }}
   :members:
   :show-inheritance:
   :member-order: bysource
{% endfor %}
{% endif %}
