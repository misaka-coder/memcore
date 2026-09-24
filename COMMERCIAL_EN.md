# Licensing Terms: AGPL-3.0 and Commercial License

MemCore is distributed under a **dual-licensing model**. You may choose between two licensing paths:

| | AGPL-3.0-only | Commercial License |
| --- | --- | --- |
| **Cost** | Free | As negotiated |
| **Source Disclosure** | Required (Copyleft) | None |
| **Use Cases** | Open-source projects, internal company use, copyleft-compatible software | Closed-source distribution, SaaS, commercial software with strict compliance needs |
| **How to Obtain** | See [LICENSE](LICENSE) | Contact copyright holder |

## Understanding AGPL-3.0

The [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.txt) is an OSI-approved free software license. Under its terms, you are free to use, modify, and distribute MemCore, including for commercial purposes.

The key distinction of AGPL compared to standard copyleft licenses is **Section 13 (Remote Network Interaction)**: If you modify MemCore and offer it over a network as a service (SaaS) to external users, you must make the corresponding source code of your modified version available to those users. A typical example is deploying a modified MemCore as a multi-tenant memory API.

Common clarifications:

- **AGPL allows commercial use.** It mandates source availability, not free-of-charge redistribution. You may charge for your services.
- **AGPL does not automatically infect your entire application stack.** The copyleft boundary applies to MemCore itself and derived works directly incorporating its code. Your proprietary application code, persona instructions, domain rules, and external services interact with MemCore via standard dependency injection interfaces (`LLMClient`, `EmbeddingProvider`, `MemoryStore`, `VectorIndex`). Whether a combined work constitutes a derivative work depends on legal specifics; consult your legal counsel for definitive guidance.
- **Purely internal use does not trigger disclosure.** If MemCore is deployed within your organization solely for internal employees and not made available to external users as a service, source disclosure obligations under Section 13 are generally not triggered.

## When Do You Need a Commercial License?

A Commercial License is designed for organizations that wish to use MemCore without the copyleft obligations of AGPL-3.0, including:

- Embedding MemCore into closed-source commercial software distributed to customers (via binaries, containers, or on-premises deployments);
- Providing SaaS or cloud APIs based on MemCore without releasing modified source code under Section 13;
- Corporate legal departments that maintain strict policies prohibiting AGPL software in commercial products;
- Organizations requiring enterprise warranties, dedicated technical support, customized licensing terms, or long-term maintenance SLAs.

## How to Inquire

Commercial license terms, scopes, and fees are negotiated on a case-by-case basis. Please reach out to the copyright holder (misaka) outlining your intended use case: product architecture, whether it involves external SaaS or binary distribution, and whether source modifications are expected.

## Contributions and Copyright

Because MemCore is offered under a dual license, the copyright holder must retain the legal ability to distribute the codebase under alternative terms. If you wish to have non-trivial code contributions merged into the upstream repository, you may be asked to sign a Contributor License Agreement (CLA) or confirm that your contribution is licensed under the same dual terms.

The codebase is copyrighted by misaka.

## Related Projects

Host companion projects (such as AkaneCompanionLab) may use more permissive licenses (e.g., Apache 2.0). These licenses are independent. MemCore's AGPL obligations do not automatically apply to independent companion applications communicating over clean process or network boundaries, but **directly linking or embedding MemCore source code into a single binary will invoke copyleft terms**. Please verify your integration architecture prior to deployment.
